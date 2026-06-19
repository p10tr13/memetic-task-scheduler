"""Optuna hyperparameter tuning for one EA configuration.

Tunes the search hyperparameters (crossover/mutation probabilities, population,
tournament size, and - for memetic configs - CPM local-search attempts) of a
single configuration, then writes a ``best_params.json`` that
``run_single``/``run_campaign --params-json`` consume for the tuned-vs-default
comparison.

The objective is the mean normalized gap to the per-instance lower bound over the
held-out tuning instances (:data:`instances.TUNE_INSTANCES`) and a few seeds,
evaluated under the SAME objective-evaluation budget as the equal-budget campaign
(:data:`run_single.EQUAL_BUDGET_EVALS`) so tuning is budget-fair: a trial cannot
"win" simply by spending more evaluations.

Single process::

    python -m src.experiments.tune --crossover pmx --cpm --n-trials 150 \
        --storage results/tune/pmx_cpm.log --out results/tune/pmx_cpm.json

Many cores: launch several workers against one shared journal storage, then
summarise once (each worker only adds trials; ``--no-summary`` avoids racing on
the output file)::

    for i in $(seq 32); do \
      python -m src.experiments.tune --crossover pmx --cpm --n-trials 5 \
        --storage results/tune/pmx_cpm.log --study-name pmx_cpm --no-summary & \
    done; wait
    python -m src.experiments.tune --crossover pmx --cpm --summarize-only \
      --storage results/tune/pmx_cpm.log --study-name pmx_cpm --out results/tune/pmx_cpm.json
"""

import argparse

import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

from src.experiments import layout, run_single
from src.experiments.instances import (
    TUNE_INSTANCES,
    VALIDATION_INSTANCES,
    get_instance,
)
from src.experiments.run_baselines import _lower_bound
from src.solvers.evolutionary.algorithm import run_evolution
from src.solvers.evolutionary.config import EvolutionConfig

DEFAULT_N_TRIALS = 150
DEFAULT_TUNE_SEEDS = 3
DEFAULT_SAMPLER_SEED = 0


def main(argv: list[str] | None = None) -> None:
    """Runs (or summarises) an Optuna study and writes the tuned parameters."""
    args = _parse_arguments(argv)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = _create_study(args)

    if not args.summarize_only:
        instances = list(args.instances) if args.instances else list(TUNE_INSTANCES)
        budget = _budget(args)
        references = {
            name: _lower_bound(get_instance(name).problem_instance)
            for name in instances
        }
        study.optimize(
            lambda trial: _objective(trial, args, instances, references, budget),
            n_trials=args.n_trials,
            n_jobs=args.n_jobs,
        )

    if not args.no_summary:
        _write_best_params(study, args)


def _objective(
    trial: optuna.Trial,
    args: argparse.Namespace,
    instances: list[str],
    references: dict[str, int],
    budget: int,
) -> float:
    """Returns the mean normalized gap to the lower bound over the tune set.

    The gap of each (instance, seed) run is averaged; the running mean is reported
    per instance so a clearly-bad trial can be pruned before all instances run.
    """
    params = _suggest_params(trial, args.cpm)
    strategies = run_single.build_strategies(
        args.crossover,
        args.cpm,
        params.get("cpm_max_attempts", run_single.LOCAL_SEARCH_ATTEMPTS),
    )

    gaps: list[float] = []
    for step, name in enumerate(instances):
        entry = get_instance(name)
        reference = references[name]
        for seed in range(args.tune_seeds):
            config = EvolutionConfig(
                population_size=params["population_size"],
                generations=run_single.EQUAL_BUDGET_GENERATION_CAP,
                crossover_probability=params["crossover_probability"],
                mutation_probability=params["mutation_probability"],
                tournament_size=params["tournament_size"],
                max_evaluations=budget,
                random_seed=seed,
            )
            result = run_evolution(entry.problem_instance, config, strategies)
            gaps.append((result.best_makespan - reference) / reference)

        trial.report(sum(gaps) / len(gaps), step=step)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return sum(gaps) / len(gaps)


def _suggest_params(trial: optuna.Trial, use_cpm: bool) -> dict:
    """Samples the EA hyperparameter search space for one trial."""
    params = {
        "crossover_probability": trial.suggest_float(
            "crossover_probability", 0.6, 0.95
        ),
        "mutation_probability": trial.suggest_float("mutation_probability", 0.05, 0.5),
        "population_size": trial.suggest_int("population_size", 20, 100, step=10),
        "tournament_size": trial.suggest_int("tournament_size", 2, 7),
    }
    if use_cpm:
        params["cpm_max_attempts"] = trial.suggest_int("cpm_max_attempts", 5, 60)
    return params


def _budget(args: argparse.Namespace) -> int:
    """Returns the objective-evaluation budget used for every trial."""
    if args.max_evaluations is not None:
        return args.max_evaluations
    if args.quick:
        return run_single.QUICK_POPULATION * (run_single.QUICK_GENERATIONS + 1)
    return run_single.EQUAL_BUDGET_EVALS


def _create_study(args: argparse.Namespace) -> optuna.Study:
    """Creates or loads the study, on a shared journal storage when requested."""
    storage = _make_storage(args.storage) if args.storage else None
    return optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=args.sampler_seed),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=1),
    )


def _make_storage(path: str):
    """Builds an Optuna storage from a URL string or a journal-file path.

    A value with a scheme (e.g. ``sqlite:///x.db``) is passed through; any other
    value is treated as a journal-file path, which is process-safe for the
    multi-worker recipe in the module docstring.
    """
    if "://" in path:
        return path
    return JournalStorage(JournalFileBackend(path))


def _write_best_params(study: optuna.Study, args: argparse.Namespace) -> None:
    """Writes the tuned parameters and study metadata to ``--out``."""
    payload = {
        "crossover": args.crossover,
        "use_cpm": args.cpm,
        "params": study.best_params,
        "best_value": study.best_value,
        "max_evaluations": _budget(args),
        "tune_seeds": args.tune_seeds,
        "tune_instances": (
            list(args.instances) if args.instances else list(TUNE_INSTANCES)
        ),
        "validation_instances": list(VALIDATION_INSTANCES),
        "completed_trials": len(
            [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        ),
        "optuna_version": optuna.__version__,
    }
    layout.write_json(args.out, payload)
    label = f"{args.crossover}{'_cpm' if args.cpm else ''}"
    print(f"[tune] {label} best={study.best_value:.4f} -> {args.out}")


def _parse_arguments(argv: list[str] | None) -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Tune one EA configuration with Optuna."
    )
    parser.add_argument("--crossover", required=True, choices=("ox", "pmx"))
    parser.add_argument(
        "--cpm",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Tune the memetic (CPM) configuration instead of the pure EA.",
    )
    parser.add_argument("--n-trials", type=int, default=DEFAULT_N_TRIALS)
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="In-process Optuna concurrency. For many cores prefer several "
        "processes against one --storage (see module docstring).",
    )
    parser.add_argument("--tune-seeds", type=int, default=DEFAULT_TUNE_SEEDS)
    parser.add_argument("--sampler-seed", type=int, default=DEFAULT_SAMPLER_SEED)
    parser.add_argument(
        "--instances",
        nargs="+",
        default=None,
        help="Tuning instances (default: instances.TUNE_INSTANCES).",
    )
    parser.add_argument(
        "--max-evaluations",
        type=int,
        default=None,
        help="Objective-evaluation budget per trial run (default: equal-budget).",
    )
    parser.add_argument(
        "--storage", default=None, help="Journal-file path or storage URL."
    )
    parser.add_argument("--study-name", default=None, help="Study name in the storage.")
    parser.add_argument(
        "--out",
        type=layout.Path,
        default=None,
        help="Path for the tuned best_params.json.",
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Do not write best_params.json (for parallel workers).",
    )
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="Skip optimisation; just write best_params.json from the study.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Tiny budget for smoke tests.",
    )
    args = parser.parse_args(argv)
    if args.out is None and not args.no_summary:
        parser.error("--out is required unless --no-summary is set.")
    return args


if __name__ == "__main__":
    main()
