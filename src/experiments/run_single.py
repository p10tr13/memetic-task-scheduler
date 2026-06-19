"""Single experiment run: one instance under one EA configuration.

Runs the memetic / EA algorithm on a single named instance across a seed range,
with every parameter given on the command line, and writes a self-contained
result *cell* under a campaign directory::

    python -m src.experiments.run_single --instance syn_t15_p010 --crossover ox --cpm

EA hyperparameters default to documented global values (see ``DEFAULT_*`` below);
override any with ``--population`` / ``--generations`` / ``--crossover-prob`` /
``--mutation-prob`` / ``--seeds``. The cell ``<campaign>/runs/<instance>__<label>/``
holds config.json, runs.csv (per-seed best makespan), convergence.csv (per-seed
curves) and timing.json. Re-running the same (instance, method) overwrites only
that cell; nothing else in the campaign is touched.
"""

import argparse
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

from src.experiments import layout
from src.experiments.instances import InstanceEntry, get_instance
from src.solvers.evolutionary.algorithm import run_evolution
from src.solvers.evolutionary.config import EvolutionConfig, EvolutionStrategies
from src.solvers.evolutionary.crossovers import ox_crossover, pmx_crossover
from src.solvers.evolutionary.local_search import cpm_local_search
from src.solvers.evolutionary.strategies import swap_mutation

LOCAL_SEARCH_ATTEMPTS = 20
DEFAULT_TOURNAMENT_SIZE = 3
SCHEMA_VERSION = 2
_CROSSOVERS = {"ox": ox_crossover, "pmx": pmx_crossover}

# Maps tuned-parameter JSON keys (as written by ``src.experiments.tune``) onto the
# ``run_single`` argparse attributes they override, so a tuned ``best_params.json``
# can be applied with ``--params-json`` without bespoke parsing here.
_PARAMS_JSON_OVERRIDES = {
    "population_size": "population",
    "crossover_probability": "crossover_prob",
    "mutation_probability": "mutation_prob",
    "tournament_size": "tournament_size",
    "cpm_max_attempts": "local_search_attempts",
}


@dataclass
class _CellRun:
    """Per-cell execution outputs, including both wall and CPU timings.

    Wall seconds (``time.perf_counter``) include time the process spent waiting
    for a CPU, so they inflate when cells run concurrently (``--jobs`` > 1). CPU
    seconds (``time.process_time``) count only on-core time and are the
    parallelism-robust per-seed cost metric; compare cell costs with those, and
    use a ``--jobs 1`` run for wall-clock cost claims.
    """

    run_rows: list[dict] = field(default_factory=list)
    convergence_rows: list[dict] = field(default_factory=list)
    convergence_eval_rows: list[dict] = field(default_factory=list)
    per_seed_wall_seconds: list[float] = field(default_factory=list)
    per_seed_cpu_seconds: list[float] = field(default_factory=list)
    started_utc: str = ""


# Default EA hyperparameters. These are GLOBAL, not per-instance: a benchmark
# instance is a fixed problem, while the search budget is an experiment knob.
# The values are standard for genetic / memetic algorithms on permutation
# (scheduling) problems; override any of them on the command line.
#   crossover 0.8 - within the classic 0.6 (De Jong & Spears 1990) .. 0.9
#                   (Grefenstette 1986) range; DAG-scheduling GAs use 0.65-0.95.
#   mutation 0.3  - probability of applying swap mutation PER INDIVIDUAL (not the
#                   per-bit 0.001-0.01 of bit-string GAs); permutation GAs 0.1-0.5.
#   population 50 - De Jong & Spears (50); common range 30-100.
#   generations 100 - fixed budget (the seed-averaged convergence plots need a
#                   fixed length); memetic search converges within it.
#   seeds 15      - independent runs; below the usual 30-run convention, kept
#                   modest to bound wall-clock (a documented limitation).
DEFAULT_POPULATION = 50
DEFAULT_GENERATIONS = 100
DEFAULT_CROSSOVER_PROB = 0.8
DEFAULT_MUTATION_PROB = 0.3
DEFAULT_SEEDS = 15

# Equal objective-evaluation budget (number of schedule decodes a pure EA spends
# at the default budget: the initial population plus one offspring evaluation per
# generation). ``--equal-budget`` compares every configuration at this SAME total
# number of objective evaluations rather than the same number of generations, so
# the memetic configs - which decode ~20-40x more per generation inside CPM local
# search - no longer get a hidden budget advantage. Derived from the canonical
# defaults (NOT --population), so the budget is identical across configs even when
# a tuned config uses a different population size.
EQUAL_BUDGET_EVALS = DEFAULT_POPULATION * (DEFAULT_GENERATIONS + 1)
# With the budget as the binding stop, generations must not cut a run short, so
# they are raised to a cap far above what any config reaches within the budget.
EQUAL_BUDGET_GENERATION_CAP = 100_000

# Tiny budget for `--quick` smoke tests only (NOT a tuned hyperparameter claim).
QUICK_POPULATION = 20
QUICK_GENERATIONS = 8
QUICK_SEEDS = 3

RUN_FIELDS = [
    "instance",
    "config",
    "crossover",
    "use_cpm",
    "seed",
    "best_makespan",
    "evaluations",
    "processors",
    "nodes",
    "edges",
    "density",
]
CONVERGENCE_FIELDS = ["instance", "config", "seed", "generation", "best_makespan"]
CONVERGENCE_EVAL_FIELDS = [
    "instance",
    "config",
    "seed",
    "evaluations",
    "best_makespan",
]


def build_strategies(
    crossover_name: str,
    use_cpm: bool,
    local_search_attempts: int = LOCAL_SEARCH_ATTEMPTS,
) -> EvolutionStrategies:
    """Builds the strategy bundle for one configuration."""
    local_search = None
    if use_cpm:
        local_search = partial(cpm_local_search, max_attempts=local_search_attempts)
    # selection_strategy is left unset so the algorithm builds tournament
    # selection from ``EvolutionConfig.tournament_size`` - that is what lets
    # ``--tournament-size`` (and tuned params) actually take effect.
    return EvolutionStrategies(
        mutation_strategy=swap_mutation,
        crossover_strategy=_CROSSOVERS[crossover_name],
        local_search_strategy=local_search,
    )


def main(argv: list[str] | None = None) -> None:
    """Runs one instance under one EA configuration and writes its cell."""
    args = _parse_arguments(argv)
    _resolve_defaults(args)
    entry = get_instance(args.instance)
    campaign = layout.resolve_campaign_dir(args.campaign, args.name)
    if args.sub_cell:
        cell = layout.subcell_dir(campaign, args.instance, args.label, args.seed_start)
    else:
        cell = layout.cell_dir(campaign, args.instance, args.label)

    cell_run = _execute(entry, args)

    write_cell(
        cell,
        _config_payload(entry, args),
        cell_run.run_rows,
        cell_run.convergence_rows,
        _timing_payload(entry, args, cell_run),
        cell_run.convergence_eval_rows,
    )
    print(f"[{entry.name}__{args.label}] {args.seeds} seeds -> {cell}")


def _resolve_defaults(args: argparse.Namespace) -> None:
    """Fills population/generations/seeds/label and budget from the defaults."""
    if args.population is None:
        args.population = QUICK_POPULATION if args.quick else DEFAULT_POPULATION
    if args.generations is None:
        args.generations = QUICK_GENERATIONS if args.quick else DEFAULT_GENERATIONS
    if args.seeds is None:
        args.seeds = QUICK_SEEDS if args.quick else DEFAULT_SEEDS
    if args.label is None:
        args.label = layout.cell_label(args.crossover, args.cpm)

    if args.params_json is not None:
        _apply_params_json(args)

    if args.equal_budget:
        if args.max_evaluations is None:
            args.max_evaluations = (
                QUICK_POPULATION * (QUICK_GENERATIONS + 1)
                if args.quick
                else EQUAL_BUDGET_EVALS
            )
        # Let the budget be the binding stop: raise the generation cap above what
        # any configuration reaches within the budget so it never clips a run.
        args.generations = EQUAL_BUDGET_GENERATION_CAP


def _apply_params_json(args: argparse.Namespace) -> None:
    """Overrides EA hyperparameters from a tuned ``best_params.json`` file.

    The file is the artifact written by :mod:`src.experiments.tune`; only the keys
    in :data:`_PARAMS_JSON_OVERRIDES` are applied, so a study JSON carrying extra
    bookkeeping fields is safe to pass.

    Args:
        args: Parsed arguments to mutate in place.
    """
    params = layout.read_json(Path(args.params_json))
    tuned = params.get("params", params)
    for json_key, arg_name in _PARAMS_JSON_OVERRIDES.items():
        if json_key in tuned:
            setattr(args, arg_name, tuned[json_key])


def _execute(
    entry: InstanceEntry,
    args: argparse.Namespace,
) -> _CellRun:
    """Runs the EA across the seed range, returning rows and per-seed timings.

    Each seed is timed with both a wall clock (``perf_counter``) and a CPU clock
    (``process_time``) so the saved timing stays meaningful even when cells run
    concurrently; only the EA call itself is timed, not I/O or setup.
    """
    strategies = build_strategies(
        args.crossover, args.cpm, args.local_search_attempts
    )
    base_config = EvolutionConfig(
        population_size=args.population,
        generations=args.generations,
        crossover_probability=args.crossover_prob,
        mutation_probability=args.mutation_prob,
        tournament_size=args.tournament_size,
        max_evaluations=args.max_evaluations,
    )

    cell_run = _CellRun(started_utc=datetime.now(timezone.utc).isoformat())

    for seed in range(args.seed_start, args.seed_start + args.seeds):
        wall_start = time.perf_counter()
        cpu_start = time.process_time()
        result = run_evolution(
            problem_instance=entry.problem_instance,
            config=replace(base_config, random_seed=seed),
            strategies=strategies,
        )
        cell_run.per_seed_wall_seconds.append(
            round(time.perf_counter() - wall_start, 4)
        )
        cell_run.per_seed_cpu_seconds.append(round(time.process_time() - cpu_start, 4))
        cell_run.run_rows.append(
            _run_row(entry, args, seed, result.best_makespan, result.total_evaluations)
        )
        cell_run.convergence_rows.extend(
            _convergence_rows(entry, args, seed, result.generation_best_makespans)
        )
        cell_run.convergence_eval_rows.extend(
            _convergence_eval_rows(
                entry, args, seed, result.evaluation_best_makespans
            )
        )

    return cell_run


def _run_row(
    entry: InstanceEntry,
    args: argparse.Namespace,
    seed: int,
    best_makespan: int,
    evaluations: int,
) -> dict:
    """Builds one per-seed run row."""
    return {
        "instance": entry.name,
        "config": args.label,
        "crossover": args.crossover,
        "use_cpm": args.cpm,
        "seed": seed,
        "best_makespan": best_makespan,
        "evaluations": evaluations,
        "processors": entry.problem_instance.number_of_groups,
        "nodes": entry.metadata.nodes,
        "edges": entry.metadata.edges,
        "density": round(entry.metadata.density, 4),
    }


def _convergence_eval_rows(
    entry: InstanceEntry,
    args: argparse.Namespace,
    seed: int,
    eval_curve: list[tuple[int, int]],
) -> list[dict]:
    """Builds the per-seed best-vs-evaluations rows for equal-budget plots."""
    return [
        {
            "instance": entry.name,
            "config": args.label,
            "seed": seed,
            "evaluations": evaluations,
            "best_makespan": best,
        }
        for evaluations, best in eval_curve
    ]


def _convergence_rows(
    entry: InstanceEntry,
    args: argparse.Namespace,
    seed: int,
    curve: list[int],
) -> list[dict]:
    """Builds the per-generation convergence rows for one seed."""
    return [
        {
            "instance": entry.name,
            "config": args.label,
            "seed": seed,
            "generation": generation,
            "best_makespan": best,
        }
        for generation, best in enumerate(curve)
    ]


def _config_payload(entry: InstanceEntry, args: argparse.Namespace) -> dict:
    """Builds the cell's config.json payload."""
    return {
        "schema_version": SCHEMA_VERSION,
        "instance": entry.name,
        "label": args.label,
        "crossover": args.crossover,
        "use_cpm": args.cpm,
        "population": args.population,
        "generations": args.generations,
        "crossover_prob": args.crossover_prob,
        "mutation_prob": args.mutation_prob,
        "tournament_size": args.tournament_size,
        "local_search_attempts": args.local_search_attempts,
        "max_evaluations": args.max_evaluations,
        "equal_budget": args.equal_budget,
        "params_json": args.params_json,
        "seed_start": args.seed_start,
        "seeds": args.seeds,
        "processors": entry.problem_instance.number_of_groups,
        "nodes": entry.metadata.nodes,
        "edges": entry.metadata.edges,
        "density": round(entry.metadata.density, 4),
    }


def _timing_payload(
    entry: InstanceEntry,
    args: argparse.Namespace,
    cell_run: _CellRun,
) -> dict:
    """Builds the cell's timing.json payload.

    Records wall and CPU seconds separately. ``*_seconds`` are wall clock (kept
    under the original names); ``*_cpu_seconds`` count only on-core time and stay
    comparable across cells even when the campaign runs cells concurrently.
    """
    wall = cell_run.per_seed_wall_seconds
    cpu = cell_run.per_seed_cpu_seconds
    return {
        "schema_version": SCHEMA_VERSION,
        "instance": entry.name,
        "label": args.label,
        "per_seed_seconds": wall,
        "total_seconds": round(sum(wall), 4),
        "mean_seconds": round(sum(wall) / len(wall), 4) if wall else 0.0,
        "per_seed_cpu_seconds": cpu,
        "total_cpu_seconds": round(sum(cpu), 4),
        "mean_cpu_seconds": round(sum(cpu) / len(cpu), 4) if cpu else 0.0,
        "started_utc": cell_run.started_utc,
    }


def write_cell(
    cell: Path,
    config_payload: dict,
    run_rows: list[dict],
    convergence_rows: list[dict],
    timing_payload: dict,
    convergence_eval_rows: list[dict] | None = None,
) -> None:
    """Writes the artifacts of one result cell, overwriting any prior cell."""
    cell.mkdir(parents=True, exist_ok=True)
    layout.write_json(cell / "config.json", config_payload)
    layout.write_csv(cell / "runs.csv", RUN_FIELDS, run_rows)
    layout.write_csv(cell / "convergence.csv", CONVERGENCE_FIELDS, convergence_rows)
    layout.write_csv(
        cell / "convergence_evals.csv",
        CONVERGENCE_EVAL_FIELDS,
        convergence_eval_rows or [],
    )
    layout.write_json(cell / "timing.json", timing_payload)


def _parse_arguments(argv: list[str] | None) -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run one instance under one EA configuration."
    )
    parser.add_argument("--instance", required=True, help="Registry instance name.")
    parser.add_argument(
        "--crossover",
        required=True,
        choices=("ox", "pmx"),
        help="Crossover operator.",
    )
    parser.add_argument(
        "--cpm",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable CPM local search (memetic) instead of a pure EA.",
    )
    parser.add_argument(
        "--population",
        type=int,
        default=None,
        help="Population size (default: instance suggestion).",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=None,
        help="Number of generations (default: instance suggestion).",
    )
    parser.add_argument(
        "--crossover-prob",
        type=float,
        default=DEFAULT_CROSSOVER_PROB,
        help="Crossover probability.",
    )
    parser.add_argument(
        "--mutation-prob",
        type=float,
        default=DEFAULT_MUTATION_PROB,
        help="Mutation probability.",
    )
    parser.add_argument(
        "--tournament-size",
        type=int,
        default=DEFAULT_TOURNAMENT_SIZE,
        help="Tournament size for parent selection.",
    )
    parser.add_argument(
        "--local-search-attempts",
        type=int,
        default=LOCAL_SEARCH_ATTEMPTS,
        help="CPM local-search probes per individual (memetic configs only).",
    )
    parser.add_argument(
        "--max-evaluations",
        type=int,
        default=None,
        help="Objective-evaluation budget; stops at the generation boundary "
        "reaching it. See also --equal-budget.",
    )
    parser.add_argument(
        "--equal-budget",
        action="store_true",
        help="Compare configs at the pure-EA objective-evaluation budget "
        f"({EQUAL_BUDGET_EVALS} decodes) instead of equal generations.",
    )
    parser.add_argument(
        "--params-json",
        default=None,
        help="Path to a tuned best_params.json whose values override the "
        "population/probabilities/tournament/CPM-attempts.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=None,
        help="Number of seeds to run (default: instance suggestion).",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=0,
        help="First seed value (for disjoint, resumable ranges).",
    )
    parser.add_argument(
        "--sub-cell",
        action="store_true",
        help="Write into <cell>/seeds/<seed-start>/ for per-seed parallel "
        "campaigns (reassembled by run_campaign's merge step).",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use the tiny smoke-test budget for any unset population/generations/seeds.",
    )
    layout.add_campaign_args(parser, with_name=True)
    parser.add_argument(
        "--label",
        default=None,
        help="Override the cell label (default: <crossover>[_cpm]).",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    main()
