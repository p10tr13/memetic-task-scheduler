"""Baseline list-scheduling heuristics."""

import heapq
from dataclasses import dataclass

import networkx as nx

from src.core.decoder import calculate_makespan
from src.core.models import ProblemInstance


@dataclass(frozen=True)
class HeuristicResult:
    """Stores a heuristic scheduling result.

    Args:
        task_permutation: Precedence-valid task order produced by the heuristic.
        makespan: Makespan calculated by the list scheduling decoder.
    """

    task_permutation: list[int]
    makespan: int


def solve_hlfet(problem_instance: ProblemInstance) -> HeuristicResult:
    """Solves a scheduling instance with the HLFET heuristic.

    Args:
        problem_instance: Scheduling problem instance.

    Returns:
        HLFET task permutation and its decoded makespan.

    Raises:
        ValueError: If the graph is cyclic, task times are invalid, or processors
            are not configured.
    """
    task_permutation = create_hlfet_permutation(problem_instance)

    return HeuristicResult(
        task_permutation=task_permutation,
        makespan=calculate_makespan(task_permutation, problem_instance),
    )


def solve_mcp(problem_instance: ProblemInstance) -> HeuristicResult:
    """Solves a scheduling instance with the MCP heuristic.

    Args:
        problem_instance: Scheduling problem instance.

    Returns:
        MCP task permutation and its decoded makespan.

    Raises:
        ValueError: If the graph is cyclic, task times are invalid, or processors
            are not configured.
    """
    task_permutation = create_mcp_permutation(problem_instance)

    return HeuristicResult(
        task_permutation=task_permutation,
        makespan=calculate_makespan(task_permutation, problem_instance),
    )


def create_hlfet_permutation(problem_instance: ProblemInstance) -> list[int]:
    """Creates a Highest Level First with Estimated Times task permutation.

    Args:
        problem_instance: Scheduling problem instance.

    Returns:
        Precedence-valid task permutation.

    Raises:
        ValueError: If the graph is cyclic or task times do not cover graph tasks.
    """
    graph = problem_instance.priority_relationship

    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("HLFET requires a directed acyclic graph.")

    task_numbers = list(nx.topological_sort(graph))
    _validate_task_time_indexes(problem_instance, set(task_numbers))
    task_levels = _calculate_hlfet_levels(problem_instance, task_numbers)

    remaining_predecessor_counts = {
        task_number: graph.in_degree(task_number) for task_number in task_numbers
    }
    ready_tasks = [
        (-task_levels[task_number], task_number)
        for task_number in task_numbers
        if remaining_predecessor_counts[task_number] == 0
    ]
    heapq.heapify(ready_tasks)

    task_permutation: list[int] = []
    while ready_tasks:
        _, task_number = heapq.heappop(ready_tasks)
        task_permutation.append(task_number)

        for successor_number in graph.successors(task_number):
            remaining_predecessor_counts[successor_number] -= 1
            if remaining_predecessor_counts[successor_number] == 0:
                heapq.heappush(
                    ready_tasks,
                    (-task_levels[successor_number], successor_number),
                )

    return task_permutation


def create_mcp_permutation(problem_instance: ProblemInstance) -> list[int]:
    """Creates a Modified Critical Path task permutation.

    Args:
        problem_instance: Scheduling problem instance.

    Returns:
        Precedence-valid task permutation.

    Raises:
        ValueError: If the graph is cyclic or task times do not cover graph tasks.
    """
    graph = problem_instance.priority_relationship

    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("MCP requires a directed acyclic graph.")

    task_numbers = list(nx.topological_sort(graph))
    _validate_task_time_indexes(problem_instance, set(task_numbers))
    latest_start_times = _calculate_mcp_latest_start_times(
        problem_instance,
        task_numbers,
    )

    remaining_predecessor_counts = {
        task_number: graph.in_degree(task_number) for task_number in task_numbers
    }
    ready_tasks = [
        (latest_start_times[task_number], task_number)
        for task_number in task_numbers
        if remaining_predecessor_counts[task_number] == 0
    ]
    heapq.heapify(ready_tasks)

    task_permutation: list[int] = []
    while ready_tasks:
        _, task_number = heapq.heappop(ready_tasks)
        task_permutation.append(task_number)

        for successor_number in graph.successors(task_number):
            remaining_predecessor_counts[successor_number] -= 1
            if remaining_predecessor_counts[successor_number] == 0:
                heapq.heappush(
                    ready_tasks,
                    (latest_start_times[successor_number], successor_number),
                )

    return task_permutation


def _calculate_hlfet_levels(
    problem_instance: ProblemInstance,
    task_numbers: list[int],
) -> dict[int, int]:
    """Calculates HLFET levels for every task.

    Args:
        problem_instance: Scheduling problem instance.
        task_numbers: Topologically sorted task numbers.

    Returns:
        Task levels keyed by task number.
    """
    graph = problem_instance.priority_relationship
    task_levels: dict[int, int] = {}

    for task_number in reversed(task_numbers):
        successor_level = max(
            (
                task_levels[successor_number]
                for successor_number in graph.successors(task_number)
            ),
            default=0,
        )
        task_levels[task_number] = (
            int(problem_instance.task_times[task_number]) + successor_level
        )

    return task_levels


def _calculate_mcp_latest_start_times(
    problem_instance: ProblemInstance,
    task_numbers: list[int],
) -> dict[int, int]:
    """Calculates latest start times used by MCP priorities.

    Args:
        problem_instance: Scheduling problem instance.
        task_numbers: Topologically sorted task numbers.

    Returns:
        Latest start times keyed by task number.
    """
    graph = problem_instance.priority_relationship
    critical_path_length = _calculate_critical_path_length(
        problem_instance,
        task_numbers,
    )
    latest_finish_times: dict[int, int] = {}
    latest_start_times: dict[int, int] = {}

    for task_number in reversed(task_numbers):
        latest_finish_times[task_number] = min(
            (
                latest_start_times[successor_number]
                for successor_number in graph.successors(task_number)
            ),
            default=critical_path_length,
        )
        latest_start_times[task_number] = latest_finish_times[task_number] - int(
            problem_instance.task_times[task_number]
        )

    return latest_start_times


def _calculate_critical_path_length(
    problem_instance: ProblemInstance,
    task_numbers: list[int],
) -> int:
    """Calculates the critical path length of the precedence graph.

    Args:
        problem_instance: Scheduling problem instance.
        task_numbers: Topologically sorted task numbers.

    Returns:
        Critical path length.
    """
    graph = problem_instance.priority_relationship
    earliest_finish_times: dict[int, int] = {}

    for task_number in task_numbers:
        earliest_start_time = max(
            (
                earliest_finish_times[predecessor_number]
                for predecessor_number in graph.predecessors(task_number)
            ),
            default=0,
        )
        earliest_finish_times[task_number] = earliest_start_time + int(
            problem_instance.task_times[task_number]
        )

    return max(earliest_finish_times.values(), default=0)


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

    task_time_count = len(problem_instance.task_times)
    for task_number in graph_task_numbers:
        if task_number < 0 or task_number >= task_time_count:
            raise ValueError("Task times must contain an entry for every graph task.")
