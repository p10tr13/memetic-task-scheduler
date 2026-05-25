"""Critical Path Method utilities for task graphs."""

import networkx as nx

from src.core.models import ProblemInstance


def identify_critical_tasks(problem_instance: ProblemInstance) -> set[int]:
    """Identifies tasks with zero slack in the precedence graph.

    Args:
        problem_instance: Scheduling problem with a precedence graph and task times.

    Returns:
        Task numbers with zero slack.

    Raises:
        ValueError: If the graph is cyclic or task times do not cover graph tasks.
    """
    priority_relationship = problem_instance.priority_relationship

    if not nx.is_directed_acyclic_graph(priority_relationship):
        raise ValueError("CPM requires a directed acyclic graph.")

    task_numbers = list(nx.topological_sort(priority_relationship))
    _validate_task_time_indexes(problem_instance, set(task_numbers))

    early_start_times: dict[int, int] = {}
    early_finish_times: dict[int, int] = {}

    for task_number in task_numbers:
        early_start_times[task_number] = max(
            (
                early_finish_times[predecessor_number]
                for predecessor_number in priority_relationship.predecessors(
                    task_number
                )
            ),
            default=0,
        )
        early_finish_times[task_number] = early_start_times[task_number] + int(
            problem_instance.task_times[task_number]
        )

    project_duration = max(early_finish_times.values(), default=0)
    late_start_times: dict[int, int] = {}
    late_finish_times: dict[int, int] = {}

    for task_number in reversed(task_numbers):
        late_finish_times[task_number] = min(
            (
                late_start_times[successor_number]
                for successor_number in priority_relationship.successors(task_number)
            ),
            default=project_duration,
        )
        late_start_times[task_number] = late_finish_times[task_number] - int(
            problem_instance.task_times[task_number]
        )

    return {
        task_number
        for task_number in task_numbers
        if late_start_times[task_number] - early_start_times[task_number] == 0
    }


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

    if any(
        task_number < 0 or task_number >= len(problem_instance.task_times)
        for task_number in graph_task_numbers
    ):
        raise ValueError("Task times must contain an entry for every graph task.")
