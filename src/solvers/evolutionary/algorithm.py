"""DEAP-based evolutionary algorithm for task scheduling."""

import random

from deap import tools

from src.core.models import ProblemInstance
from src.solvers.evolutionary.config import (
    CrossoverStrategy,
    EvolutionConfig,
    EvolutionResult,
    EvolutionStrategies,
    Individual,
    LocalSearchStrategy,
    MutationStrategy,
    SelectionStrategy,
)
from src.solvers.evolutionary.deap_setup import (
    create_toolbox,
    ensure_deap_classes,
    evaluate_population,
)
from src.solvers.evolutionary.crossovers import ox_crossover, pmx_crossover
from src.solvers.evolutionary.strategies import (
    evaluate_permutation_fitness,
    null_crossover,
    null_local_search,
    swap_mutation,
    tournament_selection,
)

__all__ = [
    "EvolutionConfig",
    "EvolutionResult",
    "EvolutionStrategies",
    "evaluate_permutation_fitness",
    "null_crossover",
    "null_local_search",
    "ox_crossover",
    "pmx_crossover",
    "run_evolution",
    "swap_mutation",
    "tournament_selection",
]


def run_evolution(
    problem_instance: ProblemInstance,
    config: EvolutionConfig,
    strategies: EvolutionStrategies | None = None,
) -> EvolutionResult:
    """Runs a configurable DEAP evolutionary algorithm.

    Args:
        problem_instance: Scheduling problem instance.
        config: Evolutionary algorithm configuration.
        strategies: Optional injectable evolutionary strategy functions.

    Returns:
        Best permutation, its makespan, and per-generation best makespans.

    Raises:
        ValueError: If algorithm parameters are invalid.
    """
    _validate_evolution_config(config)
    selected_strategies = strategies or EvolutionStrategies()
    if config.random_seed is not None:
        random.seed(config.random_seed)

    ensure_deap_classes()
    toolbox = create_toolbox(problem_instance, config)
    selected_strategy, selected_mutation, selected_crossover, selected_local_search = (
        _resolve_strategies(selected_strategies, config.tournament_size)
    )

    population = getattr(toolbox, "population")(n=config.population_size)
    hall_of_fame = tools.HallOfFame(1)
    generation_best_makespans: list[int] = []

    evaluate_population(population, toolbox)
    hall_of_fame.update(population)
    generation_best_makespans.append(int(hall_of_fame[0].fitness.values[0]))

    for _ in range(config.generations):
        offspring = selected_strategy(population, len(population))
        clone = getattr(toolbox, "clone")
        offspring = [clone(individual) for individual in offspring]

        _apply_crossover(offspring, selected_crossover, config.crossover_probability)
        _apply_mutation(offspring, selected_mutation, config.mutation_probability)
        _apply_local_search(offspring, selected_local_search, problem_instance)
        evaluate_population(offspring, toolbox)

        population = offspring
        hall_of_fame.update(population)
        generation_best_makespans.append(int(hall_of_fame[0].fitness.values[0]))

    return EvolutionResult(
        best_permutation=list(hall_of_fame[0]),
        best_makespan=int(hall_of_fame[0].fitness.values[0]),
        generation_best_makespans=generation_best_makespans,
    )


def _resolve_strategies(
    strategies: EvolutionStrategies,
    tournament_size: int,
) -> tuple[SelectionStrategy, MutationStrategy, CrossoverStrategy, LocalSearchStrategy]:
    """Resolves configured strategies to concrete callables.

    Args:
        strategies: Optional strategy callables.
        tournament_size: Tournament size for default selection.

    Returns:
        Selection, mutation, crossover, and local search callables.
    """
    selection_strategy = strategies.selection_strategy or (
        lambda population, selection_size: tournament_selection(
            population,
            selection_size,
            tournament_size,
        )
    )

    return (
        selection_strategy,
        strategies.mutation_strategy or swap_mutation,
        strategies.crossover_strategy or null_crossover,
        strategies.local_search_strategy or null_local_search,
    )


def _apply_crossover(
    offspring: list[Individual],
    crossover_strategy: CrossoverStrategy,
    crossover_probability: float,
) -> None:
    """Applies crossover to offspring pairs.

    Args:
        offspring: Offspring population to modify.
        crossover_strategy: Crossover callable.
        crossover_probability: Probability of applying crossover.
    """
    for first_child, second_child in zip(offspring[::2], offspring[1::2]):
        if random.random() >= crossover_probability:
            continue
        crossover_strategy(first_child, second_child)
        del first_child.fitness.values
        del second_child.fitness.values


def _apply_mutation(
    offspring: list[Individual],
    mutation_strategy: MutationStrategy,
    mutation_probability: float,
) -> None:
    """Applies mutation to offspring.

    Args:
        offspring: Offspring population to modify.
        mutation_strategy: Mutation callable.
        mutation_probability: Probability of applying mutation.
    """
    for individual in offspring:
        if random.random() >= mutation_probability:
            continue
        mutation_strategy(individual)
        del individual.fitness.values


def _apply_local_search(
    offspring: list[Individual],
    local_search_strategy: LocalSearchStrategy,
    problem_instance: ProblemInstance,
) -> None:
    """Applies local search to offspring.

    Args:
        offspring: Offspring population to modify.
        local_search_strategy: Local search callable.
        problem_instance: Scheduling problem instance.
    """
    for individual in offspring:
        try:
            searched_permutation = local_search_strategy(problem_instance, individual)
        except ValueError:
            continue
        if list(individual) == searched_permutation:
            continue
        individual[:] = searched_permutation
        del individual.fitness.values


def _validate_evolution_config(config: EvolutionConfig) -> None:
    """Validates evolutionary algorithm configuration.

    Args:
        config: Evolutionary algorithm configuration.

    Raises:
        ValueError: If any algorithm configuration value is invalid.
    """
    if config.population_size <= 0:
        raise ValueError("Population size must be positive.")
    if config.generations < 0:
        raise ValueError("Generations cannot be negative.")
    if not 0.0 <= config.crossover_probability <= 1.0:
        raise ValueError("Crossover probability must be between 0 and 1.")
    if not 0.0 <= config.mutation_probability <= 1.0:
        raise ValueError("Mutation probability must be between 0 and 1.")
