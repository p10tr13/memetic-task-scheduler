"""App entry point."""

from dataclasses import replace
from pathlib import Path

from src.data.kasahara_parser import parse_stg_file
from src.solvers.evolutionary.algorithm import run_evolution
from src.solvers.evolutionary.config import EvolutionConfig, EvolutionStrategies
from src.solvers.evolutionary.crossovers import pmx_crossover
from src.solvers.evolutionary.local_search import cpm_local_search
from src.solvers.evolutionary.strategies import swap_mutation, tournament_selection


def main() -> None:
    """Runs the evolutionary scheduler on a bundled sample instance."""
    repository_root = Path(__file__).resolve().parents[1]
    instance_path = repository_root / "data" / "50" / "rand0000.stg"
    problem_instance = replace(
        parse_stg_file(instance_path),
        number_of_groups=4,
    )
    evolution_config = EvolutionConfig(
        population_size=50,
        generations=100,
        crossover_probability=0.8,
        mutation_probability=0.3,
        random_seed=42,
    )
    evolution_strategies = EvolutionStrategies(
        selection_strategy=tournament_selection,
        mutation_strategy=swap_mutation,
        crossover_strategy=pmx_crossover,
        local_search_strategy=lambda instance, permutation: cpm_local_search(
            instance,
            permutation,
            max_attempts=20,
        ),
    )

    result = run_evolution(
        problem_instance=problem_instance,
        config=evolution_config,
        strategies=evolution_strategies,
    )

    print(f"Instance: {instance_path}")
    print(f"Processors: {problem_instance.number_of_groups}")
    print(f"Best makespan: {result.best_makespan}")
    print(f"Best permutation: {result.best_permutation}")


if __name__ == "__main__":
    main()
