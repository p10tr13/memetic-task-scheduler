"""Full experiment campaign runner for the memetic scheduler.

Runs the OX/PMX x CPM-on/off matrix over many seeds on every benchmark instance,
computes the HLFET/MCP baselines, the exact MILP optimum (small instances) and a
lower bound, then writes raw per-run CSVs for downstream analysis and plotting.

H1 budget note: the memetic configurations perform extra makespan evaluations in
the CPM local search, so configurations are compared at equal *generations*, not
at an equal evaluation budget. This is documented as a caveat in RESULTS.md.

Run from the repository root::

    python -m src.experiments.run_experiments            # full campaign
    python -m src.experiments.run_experiments --quick    # tiny smoke test
"""

import argparse
import csv
import math
import statistics
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path

import networkx as nx

from src.core.decoder import compute_schedule
from src.core.models import ProblemInstance
from src.experiments.instances import ExperimentInstance, build_instances
from src.solvers.evolutionary.algorithm import run_evolution
from src.solvers.evolutionary.config import CrossoverStrategy, EvolutionStrategies
from src.solvers.evolutionary.crossovers import ox_crossover, pmx_crossover
from src.solvers.evolutionary.local_search import cpm_local_search
from src.solvers.evolutionary.strategies import swap_mutation, tournament_selection
from src.solvers.heuristics import solve_hlfet, solve_mcp
from src.solvers.milp import solve_milp

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "results" / "campaign"
LOCAL_SEARCH_ATTEMPTS = 20


@dataclass(frozen=True)
class _ConfigSpec:
    """One EA configuration in the comparison matrix."""

    name: str
    crossover: CrossoverStrategy
    use_cpm: bool


CONFIGS = (
    _ConfigSpec("ox", ox_crossover, False),
    _ConfigSpec("pmx", pmx_crossover, False),
    _ConfigSpec("ox_cpm", ox_crossover, True),
    _ConfigSpec("pmx_cpm", pmx_crossover, True),
)

RUN_FIELDS = [
    "instance",
    "config",
    "seed",
    "best_makespan",
    "processors",
    "nodes",
    "edges",
    "density",
]
CONVERGENCE_FIELDS = ["instance", "config", "generation", "mean_best_makespan"]
BASELINE_FIELDS = [
    "instance",
    "processors",
    "nodes",
    "edges",
    "density",
    "hlfet",
    "mcp",
    "milp_makespan",
    "milp_is_optimal",
    "lower_bound",
]


def main(argv: list[str] | None = None) -> None:
    """Runs the campaign and writes raw CSVs to the output directory."""
    arguments = _parse_arguments(argv)
    instances = build_instances(quick=arguments.quick)
    arguments.output_dir.mkdir(parents=True, exist_ok=True)

    run_rows: list[dict] = []
    convergence_rows: list[dict] = []
    baseline_rows: list[dict] = []

    for entry in instances:
        seed_count = arguments.seeds or entry.seed_count
        print(f"[{entry.name}] {seed_count} seeds x {len(CONFIGS)} configs")
        instance_runs, instance_curves = _run_instance(entry, seed_count)
        run_rows.extend(instance_runs)
        convergence_rows.extend(instance_curves)
        baseline_rows.append(_baseline_row(entry, arguments.milp_time_limit))

    _write_csv(arguments.output_dir / "runs.csv", RUN_FIELDS, run_rows)
    _write_csv(
        arguments.output_dir / "convergence.csv", CONVERGENCE_FIELDS, convergence_rows
    )
    _write_csv(arguments.output_dir / "baselines.csv", BASELINE_FIELDS, baseline_rows)
    print(f"Wrote runs/convergence/baselines to {arguments.output_dir}")


def _run_instance(
    entry: ExperimentInstance,
    seed_count: int,
) -> tuple[list[dict], list[dict]]:
    """Runs every configuration on one instance across the seed range.

    Args:
        entry: Benchmark instance to run.
        seed_count: Number of seeds (0..seed_count-1) per configuration.

    Returns:
        Per-run rows and mean-convergence rows for this instance.
    """
    run_rows: list[dict] = []
    convergence_rows: list[dict] = []

    for config in CONFIGS:
        best_makespans, mean_curve = _run_configuration(entry, config, seed_count)
        for seed, best_makespan in enumerate(best_makespans):
            run_rows.append(_run_row(entry, config.name, seed, best_makespan))
        for generation, mean_best in enumerate(mean_curve):
            convergence_rows.append(
                {
                    "instance": entry.name,
                    "config": config.name,
                    "generation": generation,
                    "mean_best_makespan": round(mean_best, 3),
                }
            )

    return run_rows, convergence_rows


def _run_configuration(
    entry: ExperimentInstance,
    config: _ConfigSpec,
    seed_count: int,
) -> tuple[list[int], list[float]]:
    """Runs one configuration across all seeds.

    Args:
        entry: Benchmark instance to run.
        config: EA configuration to apply.
        seed_count: Number of seeds to run.

    Returns:
        Best makespan per seed and the seed-averaged convergence curve.
    """
    strategies = _build_strategies(config)
    best_makespans: list[int] = []
    curves: list[list[int]] = []

    for seed in range(seed_count):
        result = run_evolution(
            problem_instance=entry.problem_instance,
            config=replace(entry.evolution_config, random_seed=seed),
            strategies=strategies,
        )
        best_makespans.append(result.best_makespan)
        curves.append(result.generation_best_makespans)

    mean_curve = [statistics.mean(values) for values in zip(*curves)]
    return best_makespans, mean_curve


def _build_strategies(config: _ConfigSpec) -> EvolutionStrategies:
    """Builds the strategy bundle for one configuration."""
    local_search = None
    if config.use_cpm:
        local_search = partial(cpm_local_search, max_attempts=LOCAL_SEARCH_ATTEMPTS)

    return EvolutionStrategies(
        selection_strategy=tournament_selection,
        mutation_strategy=swap_mutation,
        crossover_strategy=config.crossover,
        local_search_strategy=local_search,
    )


def _run_row(
    entry: ExperimentInstance,
    config_name: str,
    seed: int,
    best_makespan: int,
) -> dict:
    """Builds one per-run CSV row."""
    return {
        "instance": entry.name,
        "config": config_name,
        "seed": seed,
        "best_makespan": best_makespan,
        "processors": entry.problem_instance.number_of_groups,
        "nodes": entry.metadata.nodes,
        "edges": entry.metadata.edges,
        "density": round(entry.metadata.density, 4),
    }


def _baseline_row(entry: ExperimentInstance, milp_time_limit: float) -> dict:
    """Builds the baseline/optimum/lower-bound row for one instance."""
    milp_makespan, milp_is_optimal = _milp_optimum(entry, milp_time_limit)

    return {
        "instance": entry.name,
        "processors": entry.problem_instance.number_of_groups,
        "nodes": entry.metadata.nodes,
        "edges": entry.metadata.edges,
        "density": round(entry.metadata.density, 4),
        "hlfet": solve_hlfet(entry.problem_instance).makespan,
        "mcp": solve_mcp(entry.problem_instance).makespan,
        "milp_makespan": milp_makespan if milp_makespan is not None else "",
        "milp_is_optimal": milp_is_optimal,
        "lower_bound": _lower_bound(entry.problem_instance),
    }


def _milp_optimum(
    entry: ExperimentInstance,
    time_limit: float,
) -> tuple[int | None, bool]:
    """Returns the MILP makespan and whether it was proven optimal."""
    if not entry.supports_milp:
        return None, False

    result = solve_milp(entry.problem_instance, time_limit_seconds=time_limit)
    return result.makespan, result.is_optimal


def _lower_bound(problem_instance: ProblemInstance) -> int:
    """Computes ``max(load, critical path)`` as a makespan lower bound.

    The critical path is the makespan with unlimited processors, obtained by
    decoding a topological order with one processor per task (no resource
    contention), reusing the decoder instead of a separate longest-path routine.

    Args:
        problem_instance: Scheduling instance with a configured processor count.

    Returns:
        Lower bound on the optimal makespan.
    """
    task_times = problem_instance.task_times
    total_work = int(sum(int(task_time) for task_time in task_times))
    load_bound = math.ceil(total_work / problem_instance.number_of_groups)

    topological_order = list(
        nx.topological_sort(problem_instance.priority_relationship)
    )
    critical_path = compute_schedule(
        topological_order,
        replace(problem_instance, number_of_groups=max(len(topological_order), 1)),
    ).makespan

    return max(load_bound, critical_path)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    """Writes rows to a CSV file with the given header."""
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _parse_arguments(argv: list[str] | None) -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Run the experiment campaign.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a tiny subset with reduced budgets for smoke-testing.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=None,
        help="Override the per-instance seed count.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the output CSV files.",
    )
    parser.add_argument(
        "--milp-time-limit",
        type=float,
        default=120.0,
        help="CBC time limit (seconds) for the exact MILP on small instances.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    main()
