"""Permutation crossover operators for evolutionary scheduling."""

import random

from src.solvers.evolutionary.config import Individual


def ox_crossover(
    first_individual: Individual,
    second_individual: Individual,
) -> tuple[Individual, Individual]:
    """Applies order crossover to two task permutations.

    Args:
        first_individual: First parent permutation modified in place.
        second_individual: Second parent permutation modified in place.

    Returns:
        Two offspring permutations.

    Raises:
        ValueError: If parent permutations are incompatible.
    """
    _validate_parent_permutations(first_individual, second_individual)

    if len(first_individual) < 2:
        return first_individual, second_individual

    start_index, end_index = _select_crossover_points(len(first_individual))
    first_child = _create_ox_child(
        first_individual, second_individual, start_index, end_index
    )
    second_child = _create_ox_child(
        second_individual, first_individual, start_index, end_index
    )

    first_individual[:] = first_child
    second_individual[:] = second_child

    return first_individual, second_individual


def pmx_crossover(
    first_individual: Individual,
    second_individual: Individual,
) -> tuple[Individual, Individual]:
    """Applies partially mapped crossover to two task permutations.

    Args:
        first_individual: First parent permutation modified in place.
        second_individual: Second parent permutation modified in place.

    Returns:
        Two offspring permutations.

    Raises:
        ValueError: If parent permutations are incompatible.
    """
    _validate_parent_permutations(first_individual, second_individual)

    if len(first_individual) < 2:
        return first_individual, second_individual

    start_index, end_index = _select_crossover_points(len(first_individual))
    first_child = _create_pmx_child(
        first_individual, second_individual, start_index, end_index
    )
    second_child = _create_pmx_child(
        second_individual, first_individual, start_index, end_index
    )

    first_individual[:] = first_child
    second_individual[:] = second_child

    return first_individual, second_individual


def _create_ox_child(
    segment_parent: Individual,
    fill_parent: Individual,
    start_index: int,
    end_index: int,
) -> Individual:
    """Creates one order crossover child.

    Args:
        segment_parent: Parent donating the fixed segment.
        fill_parent: Parent donating remaining task order.
        start_index: Inclusive segment start index.
        end_index: Exclusive segment end index.

    Returns:
        Child permutation.
    """
    child: list[int | None] = [None] * len(segment_parent)
    child[start_index:end_index] = segment_parent[start_index:end_index]
    fill_values = [task_id for task_id in fill_parent if task_id not in child]
    fill_position = 0

    for child_index in list(range(end_index, len(child))) + list(range(start_index)):
        child[child_index] = fill_values[fill_position]
        fill_position += 1

    return [int(task_id) for task_id in child]


def _create_pmx_child(
    segment_parent: Individual,
    fill_parent: Individual,
    start_index: int,
    end_index: int,
) -> Individual:
    """Creates one partially mapped crossover child.

    Args:
        segment_parent: Parent donating the fixed segment.
        fill_parent: Parent donating mapped remaining tasks.
        start_index: Inclusive segment start index.
        end_index: Exclusive segment end index.

    Returns:
        Child permutation.
    """
    child: list[int | None] = [None] * len(segment_parent)
    child[start_index:end_index] = segment_parent[start_index:end_index]

    for index in range(start_index, end_index):
        mapped_task = fill_parent[index]
        if mapped_task in child:
            continue

        target_index = index
        while start_index <= target_index < end_index:
            target_task = segment_parent[target_index]
            target_index = fill_parent.index(target_task)
        child[target_index] = mapped_task

    for index, task_id in enumerate(fill_parent):
        if child[index] is None:
            child[index] = task_id

    return [int(task_id) for task_id in child]


def _select_crossover_points(permutation_size: int) -> tuple[int, int]:
    """Selects two crossover points.

    Args:
        permutation_size: Number of tasks in each permutation.

    Returns:
        Inclusive start and exclusive end index.
    """
    start_index, end_index = sorted(random.sample(range(permutation_size), 2))

    return start_index, end_index + 1


def _validate_parent_permutations(
    first_individual: Individual,
    second_individual: Individual,
) -> None:
    """Validates that crossover parents are compatible permutations.

    Args:
        first_individual: First parent permutation.
        second_individual: Second parent permutation.

    Raises:
        ValueError: If parents differ in length, task set, or uniqueness.
    """
    first_tasks = set(first_individual)
    second_tasks = set(second_individual)

    if len(first_individual) != len(second_individual):
        raise ValueError("Parent permutations must have the same length.")
    if len(first_individual) != len(first_tasks) or len(second_individual) != len(
        second_tasks
    ):
        raise ValueError("Parent permutations cannot contain duplicate tasks.")
    if first_tasks != second_tasks:
        raise ValueError("Parent permutations must contain the same tasks.")
