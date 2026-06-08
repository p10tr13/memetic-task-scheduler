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
from src.solvers.evolutionary.strategies import swap_mutation, tournament_selection

LOCAL_SEARCH_ATTEMPTS = 20
SCHEMA_VERSION = 2
_CROSSOVERS = {"ox": ox_crossover, "pmx": pmx_crossover}


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
    "processors",
    "nodes",
    "edges",
    "density",
]
CONVERGENCE_FIELDS = ["instance", "config", "seed", "generation", "best_makespan"]


def build_strategies(crossover_name: str, use_cpm: bool) -> EvolutionStrategies:
    """Builds the strategy bundle for one configuration."""
    local_search = None
    if use_cpm:
        local_search = partial(cpm_local_search, max_attempts=LOCAL_SEARCH_ATTEMPTS)
    return EvolutionStrategies(
        selection_strategy=tournament_selection,
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
    cell = layout.cell_dir(campaign, args.instance, args.label)

    cell_run = _execute(entry, args)

    write_cell(
        cell,
        _config_payload(entry, args),
        cell_run.run_rows,
        cell_run.convergence_rows,
        _timing_payload(entry, args, cell_run),
    )
    print(f"[{entry.name}__{args.label}] {args.seeds} seeds -> {cell}")


def _resolve_defaults(args: argparse.Namespace) -> None:
    """Fills population/generations/seeds/label from the global defaults."""
    if args.population is None:
        args.population = QUICK_POPULATION if args.quick else DEFAULT_POPULATION
    if args.generations is None:
        args.generations = QUICK_GENERATIONS if args.quick else DEFAULT_GENERATIONS
    if args.seeds is None:
        args.seeds = QUICK_SEEDS if args.quick else DEFAULT_SEEDS
    if args.label is None:
        args.label = layout.cell_label(args.crossover, args.cpm)


def _execute(
    entry: InstanceEntry,
    args: argparse.Namespace,
) -> _CellRun:
    """Runs the EA across the seed range, returning rows and per-seed timings.

    Each seed is timed with both a wall clock (``perf_counter``) and a CPU clock
    (``process_time``) so the saved timing stays meaningful even when cells run
    concurrently; only the EA call itself is timed, not I/O or setup.
    """
    strategies = build_strategies(args.crossover, args.cpm)
    base_config = EvolutionConfig(
        population_size=args.population,
        generations=args.generations,
        crossover_probability=args.crossover_prob,
        mutation_probability=args.mutation_prob,
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
        cell_run.run_rows.append(_run_row(entry, args, seed, result.best_makespan))
        cell_run.convergence_rows.extend(
            _convergence_rows(entry, args, seed, result.generation_best_makespans)
        )

    return cell_run


def _run_row(
    entry: InstanceEntry,
    args: argparse.Namespace,
    seed: int,
    best_makespan: int,
) -> dict:
    """Builds one per-seed run row."""
    return {
        "instance": entry.name,
        "config": args.label,
        "crossover": args.crossover,
        "use_cpm": args.cpm,
        "seed": seed,
        "best_makespan": best_makespan,
        "processors": entry.problem_instance.number_of_groups,
        "nodes": entry.metadata.nodes,
        "edges": entry.metadata.edges,
        "density": round(entry.metadata.density, 4),
    }


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
) -> None:
    """Writes the four artifacts of one result cell, overwriting any prior cell."""
    cell.mkdir(parents=True, exist_ok=True)
    layout.write_json(cell / "config.json", config_payload)
    layout.write_csv(cell / "runs.csv", RUN_FIELDS, run_rows)
    layout.write_csv(cell / "convergence.csv", CONVERGENCE_FIELDS, convergence_rows)
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
