"""Critical Path Method utilities for task graphs."""

import weakref
from collections.abc import Sequence

import networkx as nx

from src.core.decoder import Schedule, compute_schedule
from src.core.models import ProblemInstance

# Graphs already confirmed acyclic. The acyclicity check is O(V+E) and the graph
# never changes during a run, so it is hoisted out of the per-call hot path and
# done once per graph object. A WeakSet avoids leaking graphs and avoids id()
# reuse hazards. Cyclic graphs are never added, so they still raise every call.
_VALIDATED_DAGS: "weakref.WeakSet[nx.DiGraph]" = weakref.WeakSet()


def _ensure_acyclic(graph: nx.DiGraph) -> None:
    """Validates graph acyclicity once per graph object.

    Args:
        graph: Precedence graph to validate.

    Raises:
        ValueError: If the graph is not a directed acyclic graph.
    """
    if graph in _VALIDATED_DAGS:
        return
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("CPM requires a directed acyclic graph.")
    _VALIDATED_DAGS.add(graph)


def identify_critical_tasks(
    task_permutation: Sequence[int],
    problem_instance: ProblemInstance,
) -> set[int]:
    """Identifies tasks on the critical path of the decoded schedule.

    The permutation is decoded into a concrete schedule, then the critical chain
    is traced back from the tasks finishing at the makespan, following at each
    step the predecessor whose finish forces the task's start. That predecessor
    may be a precedence predecessor or the previous task on the same processor (a
    resource bottleneck), so the result reflects schedule bottlenecks rather than
    the static graph critical path.

    Args:
        task_permutation: Task order to decode and analyse.
        problem_instance: Scheduling problem with a precedence graph and times.

    Returns:
        Task numbers lying on a critical path of the decoded schedule.

    Raises:
        ValueError: If the graph is cyclic, or the permutation/times are invalid.
    """
    _ensure_acyclic(problem_instance.priority_relationship)

    schedule = compute_schedule(task_permutation, problem_instance)
    critical_tasks: set[int] = set()
    pending = [
        task_number
        for task_number, finish_time in schedule.finish_times.items()
        if finish_time == schedule.makespan
    ]

    while pending:
        task_number = pending.pop()
        if task_number in critical_tasks:
            continue
        critical_tasks.add(task_number)
        binding_predecessors = _binding_predecessors(
            problem_instance, schedule, task_number
        )
        pending.extend(binding_predecessors)

    return critical_tasks


def _binding_predecessors(
    problem_instance: ProblemInstance,
    schedule: Schedule,
    task_number: int,
) -> list[int]:
    """Finds predecessors whose finish time forces a task's start.

    Args:
        problem_instance: Scheduling problem instance.
        schedule: Decoded schedule.
        task_number: Task whose binding predecessor is sought.

    Returns:
        Precedence and/or same-processor predecessors finishing exactly when the
        task starts, or an empty list if the task starts at time zero or nothing
        binds it.
    """
    start_time = schedule.start_times[task_number]
    if start_time == 0:
        return []

    binding_predecessors = []
    for predecessor_number in problem_instance.priority_relationship.predecessors(
        task_number
    ):
        if schedule.finish_times[predecessor_number] == start_time:
            binding_predecessors.append(predecessor_number)

    machine_predecessor = schedule.machine_predecessors[task_number]
    if (
        machine_predecessor is not None
        and schedule.finish_times[machine_predecessor] == start_time
    ):
        binding_predecessors.append(machine_predecessor)

    return binding_predecessors
