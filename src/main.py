"""App entry point."""

from dataclasses import replace
from pathlib import Path

from src.data.kasahara_parser import parse_stg_file
from src.solvers.evolutionary.algorithm import run_evolution
from src.solvers.evolutionary.config import EvolutionConfig, EvolutionStrategies
from src.solvers.evolutionary.crossovers import pmx_crossover
from src.solvers.evolutionary.local_search import cpm_local_search
from src.solvers.evolutionary.strategies import swap_mutation, tournament_selection
from src.solvers.heuristics import solve_hlfet, solve_mcp
from src.solvers.milp import solve_milp


def _optimality_gap(makespan: int, optimal_makespan: int | None) -> str:
    """Formats the optimality gap of a makespan against the MILP optimum.

    Args:
        makespan: Makespan produced by a solver.
        optimal_makespan: Proven optimal makespan, or None if unavailable.

    Returns:
        Percentage gap text, or "n/a" if the optimum is unknown.
    """
    if not optimal_makespan:
        return "n/a"

    return f"{(makespan - optimal_makespan) / optimal_makespan:.1%}"


def main() -> None:
    """Runs the heuristics, the memetic EA, and the exact MILP on a small instance.

    The instance is small enough for the MILP to prove optimality, so the heuristic
    and EA makespans are reported alongside their gap to the optimum.
    """
    repository_root = Path(__file__).resolve().parents[1]
    instance_path = repository_root / "data" / "example.stg"
    problem_instance = replace(
        parse_stg_file(instance_path),
        number_of_groups=2,
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

    hlfet_result = solve_hlfet(problem_instance)
    mcp_result = solve_mcp(problem_instance)
    evolution_result = run_evolution(
        problem_instance=problem_instance,
        config=evolution_config,
        strategies=evolution_strategies,
    )
    milp_result = solve_milp(problem_instance, time_limit_seconds=60)
    optimal_makespan = milp_result.makespan

    print(f"Instance: {instance_path}")
    print(f"Processors: {problem_instance.number_of_groups}")
    print(
        f"MILP optimum: {optimal_makespan} "
        f"(status: {milp_result.status}, proven optimal: {milp_result.is_optimal})"
    )
    print(f"{'solver':<10}{'makespan':>10}{'gap':>8}")
    print(
        f"{'HLFET':<10}{hlfet_result.makespan:>10}"
        f"{_optimality_gap(hlfet_result.makespan, optimal_makespan):>8}"
    )
    print(
        f"{'MCP':<10}{mcp_result.makespan:>10}"
        f"{_optimality_gap(mcp_result.makespan, optimal_makespan):>8}"
    )
    print(
        f"{'Memetic':<10}{evolution_result.best_makespan:>10}"
        f"{_optimality_gap(evolution_result.best_makespan, optimal_makespan):>8}"
    )


if __name__ == "__main__":
    main()
