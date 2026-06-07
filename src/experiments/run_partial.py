"""Partial-results experiment runner for the memetic scheduler.

Runs a small OX-vs-PMX and CPM-on-vs-off comparison over several seeds on a
bundled instance, prints a summary, and writes convergence data to CSV. This
targets the preliminary-results checkpoint, not the full evaluation protocol
(no baselines, MILP gap, or significance tests yet).

Run from the repository root::

    python -m src.experiments.run_partial
"""

import csv
import statistics
from dataclasses import replace
from functools import partial
from pathlib import Path

from src.core.models import ProblemInstance
from src.data.kasahara_parser import parse_stg_file
from src.solvers.evolutionary.algorithm import run_evolution
from src.solvers.evolutionary.config import (
    CrossoverStrategy,
    EvolutionConfig,
    EvolutionStrategies,
    LocalSearchStrategy,
)
from src.solvers.evolutionary.crossovers import ox_crossover, pmx_crossover
from src.solvers.evolutionary.local_search import cpm_local_search
from src.solvers.evolutionary.strategies import swap_mutation, tournament_selection

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
INSTANCE_PATH = REPOSITORY_ROOT / "data" / "example.stg"
RESULTS_DIR = REPOSITORY_ROOT / "results"
NUMBER_OF_GROUPS = 3
SEEDS = tuple(range(10))
LOCAL_SEARCH_ATTEMPTS = 20

CONFIGS: tuple[tuple[str, CrossoverStrategy, bool], ...] = (
    ("ox", ox_crossover, False),
    ("pmx", pmx_crossover, False),
    ("ox_cpm", ox_crossover, True),
    ("pmx_cpm", pmx_crossover, True),
)

BASE_CONFIG = EvolutionConfig(
    population_size=40,
    generations=60,
    crossover_probability=0.8,
    mutation_probability=0.3,
)


def build_strategies(
    crossover_strategy: CrossoverStrategy,
    use_cpm: bool,
) -> EvolutionStrategies:
    """Builds the strategy bundle for one experiment configuration.

    Args:
        crossover_strategy: Crossover operator to use.
        use_cpm: Whether to enable CPM local search (memetic) or not (plain EA).

    Returns:
        Strategy bundle for `run_evolution`.
    """
    local_search: LocalSearchStrategy | None = None
    if use_cpm:
        local_search = partial(cpm_local_search, max_attempts=LOCAL_SEARCH_ATTEMPTS)

    return EvolutionStrategies(
        selection_strategy=tournament_selection,
        mutation_strategy=swap_mutation,
        crossover_strategy=crossover_strategy,
        local_search_strategy=local_search,
    )


def run_configuration(
    problem_instance: ProblemInstance,
    crossover_strategy: CrossoverStrategy,
    use_cpm: bool,
) -> tuple[list[int], list[list[int]]]:
    """Runs one configuration across every seed.

    Args:
        problem_instance: Scheduling instance to solve.
        crossover_strategy: Crossover operator to use.
        use_cpm: Whether to enable CPM local search.

    Returns:
        Best makespan per seed and the convergence curve per seed.
    """
    strategies = build_strategies(crossover_strategy, use_cpm)
    best_makespans: list[int] = []
    convergence_curves: list[list[int]] = []

    for seed in SEEDS:
        result = run_evolution(
            problem_instance=problem_instance,
            config=replace(BASE_CONFIG, random_seed=seed),
            strategies=strategies,
        )
        best_makespans.append(result.best_makespan)
        convergence_curves.append(result.generation_best_makespans)

    return best_makespans, convergence_curves


def mean_curve(convergence_curves: list[list[int]]) -> list[float]:
    """Averages convergence curves position-wise across seeds.

    Args:
        convergence_curves: One best-makespan-per-generation curve per seed.

    Returns:
        Mean best makespan at each generation index.
    """
    return [statistics.mean(values) for values in zip(*convergence_curves)]


def write_summary_csv(summary_rows: list[dict[str, float]]) -> Path:
    """Writes the per-configuration summary table to CSV.

    Args:
        summary_rows: One row per configuration with aggregate statistics.

    Returns:
        Path of the written CSV file.
    """
    summary_path = RESULTS_DIR / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as summary_file:
        writer = csv.DictWriter(
            summary_file,
            fieldnames=["config", "best", "mean", "median"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    return summary_path


def write_convergence_csv(mean_curves: dict[str, list[float]]) -> Path:
    """Writes mean convergence curves to CSV, one column per configuration.

    Args:
        mean_curves: Mean best-makespan curve keyed by configuration name.

    Returns:
        Path of the written CSV file.
    """
    convergence_path = RESULTS_DIR / "convergence.csv"
    config_names = list(mean_curves)
    generation_count = len(next(iter(mean_curves.values())))

    with convergence_path.open("w", newline="", encoding="utf-8") as convergence_file:
        writer = csv.writer(convergence_file)
        writer.writerow(["generation", *config_names])
        for generation in range(generation_count):
            writer.writerow(
                [generation, *(mean_curves[name][generation] for name in config_names)]
            )

    return convergence_path


def main() -> None:
    """Runs the partial-results comparison and writes artifacts to `results/`."""
    RESULTS_DIR.mkdir(exist_ok=True)
    problem_instance = replace(
        parse_stg_file(INSTANCE_PATH),
        number_of_groups=NUMBER_OF_GROUPS,
    )

    summary_rows: list[dict[str, float]] = []
    mean_curves: dict[str, list[float]] = {}

    print(
        f"Instance: {INSTANCE_PATH.name}  groups: {NUMBER_OF_GROUPS}  seeds: {len(SEEDS)}"
    )
    print(f"{'config':<10}{'best':>6}{'mean':>8}{'median':>8}")

    for name, crossover_strategy, use_cpm in CONFIGS:
        best_makespans, convergence_curves = run_configuration(
            problem_instance,
            crossover_strategy,
            use_cpm,
        )
        summary_rows.append(
            {
                "config": name,
                "best": min(best_makespans),
                "mean": round(statistics.mean(best_makespans), 2),
                "median": statistics.median(best_makespans),
            }
        )
        mean_curves[name] = mean_curve(convergence_curves)
        print(
            f"{name:<10}{min(best_makespans):>6}"
            f"{statistics.mean(best_makespans):>8.2f}"
            f"{statistics.median(best_makespans):>8.1f}"
        )

    summary_path = write_summary_csv(summary_rows)
    convergence_path = write_convergence_csv(mean_curves)
    print(f"Wrote {summary_path}")
    print(f"Wrote {convergence_path}")


if __name__ == "__main__":
    main()
