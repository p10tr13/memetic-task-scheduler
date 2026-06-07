"""Local search strategies for evolutionary scheduling."""

import random
from collections.abc import Sequence

from src.core.cpm_utils import identify_critical_tasks
from src.core.decoder import calculate_makespan
from src.core.models import ProblemInstance, predecessors_map


def get_left_boundary(
    problem_instance: ProblemInstance,
    task_permutation: Sequence[int],
    task_id: int,
) -> int:
    """Finds the earliest safe predecessor boundary for a task.

    Args:
        problem_instance: Scheduling problem with a precedence graph.
        task_permutation: Current task order.
        task_id: Task to move left.

    Returns:
        Maximum index of direct predecessors, or -1 if there are none.

    Raises:
        ValueError: If the task or one of its predecessors is missing.
    """
    task_indices = {
        task_number: task_index
        for task_index, task_number in enumerate(task_permutation)
    }

    if task_id not in task_indices:
        raise ValueError(f"Task {task_id} is missing from the permutation.")

    predecessors = predecessors_map(problem_instance.priority_relationship)
    predecessor_indices = []
    for predecessor_id in predecessors[task_id]:
        if predecessor_id not in task_indices:
            raise ValueError(
                f"Predecessor {predecessor_id} is missing from the permutation."
            )
        predecessor_indices.append(task_indices[predecessor_id])

    return max(predecessor_indices, default=-1)


def create_pulled_left_mutant(
    task_permutation: Sequence[int],
    task_id: int,
    target_index: int,
) -> list[int]:
    """Moves a task left in the permutation by insertion.

    Args:
        task_permutation: Current task order.
        task_id: Task to move left.
        target_index: New index for the task.

    Returns:
        Mutated task order.

    Raises:
        ValueError: If the task is missing or target index is invalid.
    """
    mutant = list(task_permutation)

    try:
        current_index = mutant.index(task_id)
    except ValueError as missing_task_error:
        raise ValueError(
            f"Task {task_id} is missing from the permutation."
        ) from missing_task_error

    if target_index < 0 or target_index >= current_index:
        raise ValueError("Target index must move the task left.")

    mutant.pop(current_index)
    mutant.insert(target_index, task_id)

    return mutant


def cpm_local_search(
    problem_instance: ProblemInstance,
    task_permutation: Sequence[int],
    max_attempts: int,
) -> list[int]:
    """Improves a permutation by pulling schedule-critical tasks left.

    Critical tasks are recomputed from the current schedule on every attempt, so
    the search follows the bottlenecks of the evolving schedule.

    Args:
        problem_instance: Scheduling problem instance.
        task_permutation: Initial task order.
        max_attempts: Maximum number of local search attempts.

    Returns:
        Final task order after accepted local search moves.
    """
    current_permutation = list(task_permutation)
    current_cmax = calculate_makespan(
        current_permutation, problem_instance, validate=False
    )

    if max_attempts <= 0:
        return current_permutation

    for _ in range(max_attempts):
        critical_tasks = _movable_critical_tasks(problem_instance, current_permutation)
        if not critical_tasks:
            break

        task_id = random.choice(critical_tasks)
        current_index = current_permutation.index(task_id)
        left_boundary = get_left_boundary(
            problem_instance,
            current_permutation,
            task_id,
        )

        if left_boundary >= current_index - 1:
            continue

        target_index = random.randint(left_boundary + 1, current_index - 1)
        mutant = create_pulled_left_mutant(
            current_permutation,
            task_id,
            target_index,
        )
        new_cmax = calculate_makespan(mutant, problem_instance, validate=False)

        if new_cmax <= current_cmax:
            current_permutation = mutant
            current_cmax = new_cmax

    return current_permutation


def _movable_critical_tasks(
    problem_instance: ProblemInstance,
    task_permutation: Sequence[int],
) -> list[int]:
    """Returns non-zero-duration critical tasks of the current schedule.

    Args:
        problem_instance: Scheduling problem instance.
        task_permutation: Current task order.

    Returns:
        Sorted critical task numbers worth moving (zero-duration tasks excluded).
    """
    movable_tasks = []
    task_indices = {
        task_number: task_index
        for task_index, task_number in enumerate(task_permutation)
    }

    for task_id in identify_critical_tasks(task_permutation, problem_instance):
        if int(problem_instance.task_times[task_id]) <= 0:
            continue

        current_index = task_indices[task_id]
        left_boundary = get_left_boundary(problem_instance, task_permutation, task_id)
        if left_boundary < current_index - 1:
            movable_tasks.append(task_id)

    return sorted(movable_tasks)
