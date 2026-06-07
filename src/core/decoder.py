"""List scheduling decoder for task permutations."""

import heapq
from collections.abc import Sequence
from dataclasses import dataclass

from src.core.models import ProblemInstance, predecessors_map


@dataclass(frozen=True)
class Schedule:
    """Stores the schedule produced by decoding a task permutation.

    Args:
        start_times: Start time per task number on its processor.
        finish_times: Finish time per task number.
        machine_predecessors: Task that previously occupied the same processor,
            or None if the task was the first one on its processor.
        makespan: Latest finish time across all tasks.
    """

    start_times: dict[int, int]
    finish_times: dict[int, int]
    machine_predecessors: dict[int, int | None]
    makespan: int


def calculate_makespan(
    task_permutation: Sequence[int],
    problem_instance: ProblemInstance,
    validate: bool = True,
) -> int:
    """Calculates makespan for a precedence-valid task permutation.

    Args:
        task_permutation: Task numbers in the order used by list scheduling.
        problem_instance: Scheduling problem with processor count, graph, and times.
        validate: Whether to validate the permutation/time indexes before decoding.
            EA inner-loop callers pass ``False`` for already-valid permutations.

    Returns:
        Makespan produced by scheduling the permutation on available processors.

    Raises:
        ValueError: If the processor count, task times, or permutation is invalid.
    """
    return compute_schedule(task_permutation, problem_instance, validate).makespan


def compute_schedule(
    task_permutation: Sequence[int],
    problem_instance: ProblemInstance,
    validate: bool = True,
) -> Schedule:
    """Decodes a precedence-valid task permutation into a concrete schedule.

    Each task is placed on the processor that frees earliest, no earlier than all
    its predecessors finish, matching the list-scheduling decoder.

    Args:
        task_permutation: Task numbers in the order used by list scheduling.
        problem_instance: Scheduling problem with processor count, graph, and times.
        validate: Whether to validate the permutation/time indexes before decoding.
            EA inner-loop callers pass ``False`` for already-valid permutations;
            the decoder still raises on precedence violations either way.

    Returns:
        Start times, finish times, per-processor predecessors, and the makespan.

    Raises:
        ValueError: If the processor count, task times, or permutation is invalid.
    """
    processor_count = _validate_processor_count(problem_instance)

    if validate:
        _validate_schedule_inputs(task_permutation, problem_instance)

    predecessors = predecessors_map(problem_instance.priority_relationship)
    return _list_schedule(
        task_permutation, problem_instance, processor_count, predecessors
    )


def _list_schedule(
    task_permutation: Sequence[int],
    problem_instance: ProblemInstance,
    processor_count: int,
    predecessors: dict[int, list[int]],
) -> Schedule:
    """Runs the list-scheduling decode loop over a validated permutation.

    Args:
        task_permutation: Task numbers in the order used by list scheduling.
        problem_instance: Scheduling problem instance.
        processor_count: Positive number of processors.
        predecessors: Precomputed task -> predecessor-list mapping.

    Returns:
        The decoded schedule.

    Raises:
        ValueError: If a task is scheduled before one of its predecessors.
    """
    processor_heap = [(0, processor_id) for processor_id in range(processor_count)]
    heapq.heapify(processor_heap)
    last_task_on_processor: dict[int, int | None] = {
        processor_id: None for processor_id in range(processor_count)
    }
    start_times: dict[int, int] = {}
    finish_times: dict[int, int] = {}
    machine_predecessors: dict[int, int | None] = {}

    for task_number in task_permutation:
        predecessor_finish_time = _predecessor_finish_time(
            predecessors, finish_times, task_number
        )

        processor_available_time, processor_id = heapq.heappop(processor_heap)
        task_start_time = max(processor_available_time, predecessor_finish_time)
        task_finish_time = task_start_time + int(
            problem_instance.task_times[task_number]
        )

        start_times[task_number] = task_start_time
        finish_times[task_number] = task_finish_time
        machine_predecessors[task_number] = last_task_on_processor[processor_id]
        last_task_on_processor[processor_id] = task_number
        heapq.heappush(processor_heap, (task_finish_time, processor_id))

    return Schedule(
        start_times=start_times,
        finish_times=finish_times,
        machine_predecessors=machine_predecessors,
        makespan=max(finish_times.values(), default=0),
    )


def _validate_schedule_inputs(
    task_permutation: Sequence[int],
    problem_instance: ProblemInstance,
) -> None:
    """Validates the permutation and task-time indexes against the graph.

    Args:
        task_permutation: Task numbers in the order used by list scheduling.
        problem_instance: Scheduling problem instance.

    Raises:
        ValueError: If the permutation or task-time indexes are invalid.
    """
    graph_task_numbers = set(problem_instance.priority_relationship.nodes)
    _validate_task_permutation(task_permutation, graph_task_numbers)
    _validate_task_time_indexes(problem_instance, graph_task_numbers)


def _predecessor_finish_time(
    predecessors: dict[int, list[int]],
    finish_times: dict[int, int],
    task_number: int,
) -> int:
    """Returns the latest finish time among a task's scheduled predecessors.

    Args:
        predecessors: Precomputed task -> predecessor-list mapping.
        finish_times: Finish times of already-scheduled tasks.
        task_number: Task whose predecessors are inspected.

    Returns:
        Latest predecessor finish time, or 0 if the task has no predecessors.

    Raises:
        ValueError: If a predecessor has not been scheduled yet.
    """
    predecessor_finish_time = 0

    for predecessor_number in predecessors[task_number]:
        if predecessor_number not in finish_times:
            raise ValueError(
                f"Task {task_number} is scheduled before predecessor "
                f"{predecessor_number}."
            )
        predecessor_finish_time = max(
            predecessor_finish_time,
            finish_times[predecessor_number],
        )

    return predecessor_finish_time


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
