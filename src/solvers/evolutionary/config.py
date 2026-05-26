"""Configuration types for evolutionary scheduling."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from src.core.models import ProblemInstance

Individual = list[int]
SelectionStrategy = Callable[[list[Individual], int], list[Individual]]
MutationStrategy = Callable[[Individual], tuple[Individual]]
CrossoverStrategy = Callable[[Individual, Individual], tuple[Individual, Individual]]
LocalSearchStrategy = Callable[[ProblemInstance, Sequence[int]], Individual]


@dataclass(frozen=True)
class EvolutionConfig:
    """Stores evolutionary algorithm configuration.

    Args:
        population_size: Number of individuals in the population.
        generations: Number of evolutionary generations.
        crossover_probability: Probability of applying crossover to a pair.
        mutation_probability: Probability of applying mutation to an individual.
        tournament_size: Tournament size used by the default selection strategy.
        invalid_makespan_penalty: Fitness assigned to invalid permutations.
        random_seed: Optional seed for reproducible runs.
    """

    population_size: int
    generations: int
    crossover_probability: float
    mutation_probability: float
    tournament_size: int = 3
    invalid_makespan_penalty: int | None = None
    random_seed: int | None = None


@dataclass(frozen=True)
class EvolutionStrategies:
    """Stores injectable evolutionary strategy functions.

    Args:
        selection_strategy: Callable used to select parents.
        mutation_strategy: Callable used to mutate an individual.
        crossover_strategy: Callable used to cross two individuals.
        local_search_strategy: Callable used to locally improve each offspring.
    """

    selection_strategy: SelectionStrategy | None = None
    mutation_strategy: MutationStrategy | None = None
    crossover_strategy: CrossoverStrategy | None = None
    local_search_strategy: LocalSearchStrategy | None = None


@dataclass(frozen=True)
class EvolutionResult:
    """Stores the result of an evolutionary scheduling run.

    Args:
        best_permutation: Best task permutation found by the algorithm.
        best_makespan: Makespan of the best task permutation.
        generation_best_makespans: Best makespan recorded after each generation.
    """

    best_permutation: Individual
    best_makespan: int
    generation_best_makespans: list[int]
