"""List scheduling decoder for task permutations."""

import heapq
from collections.abc import Sequence

import numpy as np

from src.core.models import ProblemInstance


def calculate_makespan(
    task_permutation: Sequence[int],
    problem_instance: ProblemInstance,
) -> int:
    """Calculates makespan for a precedence-valid task permutation.

    Args:
        task_permutation: Task numbers in the order used by list scheduling.
        problem_instance: Scheduling problem with processor count, graph, and times.

    Returns:
        Makespan produced by scheduling the permutation on available processors.

    Raises:
        ValueError: If the processor count, task times, or permutation is invalid.
    """
    processor_count = _validate_processor_count(problem_instance)
    graph_task_numbers = set(problem_instance.priority_relationship.nodes)

    _validate_task_permutation(task_permutation, graph_task_numbers)
    _validate_task_time_indexes(problem_instance, graph_task_numbers)

    processor_available_times = [0] * processor_count
    heapq.heapify(processor_available_times)
    task_finish_times = np.zeros(len(problem_instance.task_times), dtype=int)
    scheduled_task_numbers: set[int] = set()

    for task_number in task_permutation:
        predecessor_finish_time = 0

        for predecessor_number in problem_instance.priority_relationship.predecessors(
            task_number
        ):
            if predecessor_number not in scheduled_task_numbers:
                raise ValueError(
                    f"Task {task_number} is scheduled before predecessor "
                    f"{predecessor_number}."
                )
            predecessor_finish_time = max(
                predecessor_finish_time,
                int(task_finish_times[predecessor_number]),
            )

        processor_available_time = heapq.heappop(processor_available_times)
        task_start_time = max(processor_available_time, predecessor_finish_time)
        task_finish_time = task_start_time + int(
            problem_instance.task_times[task_number]
        )

        task_finish_times[task_number] = task_finish_time
        scheduled_task_numbers.add(task_number)
        heapq.heappush(processor_available_times, task_finish_time)

    return int(task_finish_times.max()) if task_finish_times.size else 0


def _validate_processor_count(problem_instance: ProblemInstance) -> int:
    """Validates and returns the processor count.

    Args:
        problem_instance: Scheduling problem instance.

    Returns:
        Positive number of processors.

    Raises:
        ValueError: If the number of processors is not configured or positive.
    """
    processor_count = problem_instance.number_of_groups

    if processor_count is None:
        raise ValueError("Problem instance must define the number of processors.")
    if processor_count <= 0:
        raise ValueError("Number of processors must be positive.")

    return processor_count


def _validate_task_permutation(
    task_permutation: Sequence[int],
    graph_task_numbers: set[int],
) -> None:
    """Validates the task permutation against the graph task numbers.

    Args:
        task_permutation: Task numbers in the order used by list scheduling.
        graph_task_numbers: Task numbers from the precedence graph.

    Raises:
        ValueError: If the task permutation is invalid.
    """
    permutation_task_numbers = set(task_permutation)

    if len(task_permutation) != len(permutation_task_numbers):
        raise ValueError("Task permutation contains duplicate task numbers.")

    if permutation_task_numbers != graph_task_numbers:
        missing_task_numbers = graph_task_numbers - permutation_task_numbers
        unknown_task_numbers = permutation_task_numbers - graph_task_numbers
        raise ValueError(
            "Task permutation must contain every graph task exactly once. "
            f"Missing: {sorted(missing_task_numbers)}. "
            f"Unknown: {sorted(unknown_task_numbers)}."
        )


def _validate_task_time_indexes(
    problem_instance: ProblemInstance,
    graph_task_numbers: set[int],
) -> None:
    """Validates that graph task numbers can index the task time array.

    Args:
        problem_instance: Scheduling problem instance.
        graph_task_numbers: Task numbers from the precedence graph.

    Raises:
        ValueError: If task numbers cannot index the task time array.
    """
    if not graph_task_numbers:
        return

    smallest_task_number = min(graph_task_numbers)
    largest_task_number = max(graph_task_numbers)

    if smallest_task_number < 0 or largest_task_number >= len(
        problem_instance.task_times
    ):
        raise ValueError("Task times must contain an entry for every graph task.")
