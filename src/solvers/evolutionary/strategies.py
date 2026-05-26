"""Default evolutionary strategies for task scheduling."""

import random
from collections.abc import Sequence

from deap import tools

from src.core.decoder import calculate_makespan
from src.core.models import ProblemInstance
from src.solvers.evolutionary.config import Individual


def evaluate_permutation_fitness(
    task_permutation: Sequence[int],
    problem_instance: ProblemInstance,
    invalid_makespan_penalty: int,
) -> tuple[int]:
    """Evaluates a task permutation for DEAP minimization.

    Args:
        task_permutation: Task order to evaluate.
        problem_instance: Scheduling problem instance.
        invalid_makespan_penalty: Fitness returned for invalid permutations.

    Returns:
        One-item fitness tuple containing makespan or penalty.
    """
    try:
        return (calculate_makespan(task_permutation, problem_instance),)
    except ValueError:
        return (invalid_makespan_penalty,)


def tournament_selection(
    population: list[Individual],
    selection_size: int,
    tournament_size: int = 3,
) -> list[Individual]:
    """Selects individuals using DEAP tournament selection.

    Args:
        population: Current population.
        selection_size: Number of individuals to select.
        tournament_size: Number of competitors in each tournament.

    Returns:
        Selected individuals.
    """
    return list(
        tools.selTournament(
            population,
            selection_size,
            tournsize=tournament_size,
        )
    )


def swap_mutation(individual: Individual) -> tuple[Individual]:
    """Mutates an individual by swapping two task positions.

    Args:
        individual: Task permutation to mutate in place.

    Returns:
        Mutated individual wrapped for DEAP compatibility.
    """
    if len(individual) < 2:
        return (individual,)

    first_index, second_index = random.sample(range(len(individual)), 2)
    individual[first_index], individual[second_index] = (
        individual[second_index],
        individual[first_index],
    )

    return (individual,)


def null_crossover(
    first_individual: Individual,
    second_individual: Individual,
) -> tuple[Individual, Individual]:
    """Leaves two individuals unchanged during crossover.

    Args:
        first_individual: First parent individual.
        second_individual: Second parent individual.

    Returns:
        Unchanged individuals.
    """
    return first_individual, second_individual


def null_local_search(
    problem_instance: ProblemInstance,
    task_permutation: Sequence[int],
) -> Individual:
    """Leaves an individual unchanged during local search.

    Args:
        problem_instance: Scheduling problem instance.
        task_permutation: Task order to return.

    Returns:
        Copy of the input task permutation.
    """
    _ = problem_instance
    return list(task_permutation)
