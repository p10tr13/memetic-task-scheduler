"""Configuration types for evolutionary scheduling."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from src.core.models import ProblemInstance

Individual = list[int]
SelectionStrategy = Callable[[list[Individual], int], list[Individual]]
MutationStrategy = Callable[[Individual], tuple[Individual]]
CrossoverStrategy = Callable[[Individual, Individual], tuple[Individual, Individual]]
LocalSearchStrategy = Callable[[ProblemInstance, Sequence[int]], Individual]


@dataclass(frozen=True)
class EvolutionConfig:  # pylint: disable=too-many-instance-attributes
    """Stores evolutionary algorithm configuration.

    Args:
        population_size: Number of individuals in the population.
        generations: Number of evolutionary generations.
        crossover_probability: Probability of applying crossover to a pair.
        mutation_probability: Probability of applying mutation to an individual.
        tournament_size: Tournament size used by the default selection strategy.
        invalid_makespan_penalty: Fitness assigned to invalid permutations.
        max_evaluations: Optional objective-evaluation budget. When set, evolution
            stops at the first generation boundary where the cumulative number of
            schedule decodes reaches this budget, so configurations that decode
            more per generation (e.g. CPM local search) get the same total budget
            as a pure EA. ``generations`` still caps the run when the budget is not
            reached first.
        random_seed: Optional seed for reproducible runs.
    """

    population_size: int
    generations: int
    crossover_probability: float
    mutation_probability: float
    tournament_size: int = 3
    invalid_makespan_penalty: int | None = None
    max_evaluations: int | None = None
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
        total_evaluations: Total number of schedule decodes performed across the
            whole run (initial population, per-generation evaluation, and every
            CPM local-search probe). 0 unless evolution tracked it.
        evaluation_best_makespans: ``(cumulative_evaluations, best_makespan)``
            samples recorded at each generation boundary, for plotting convergence
            against the objective-evaluation budget rather than the generation
            index.
    """

    best_permutation: Individual
    best_makespan: int
    generation_best_makespans: list[int]
    total_evaluations: int = 0
    evaluation_best_makespans: list[tuple[int, int]] = field(default_factory=list)
