"""Exact MILP solver for small task scheduling instances.

Time-indexed formulation of P|prec|C_max on identical processors, solved with
CBC via PuLP. Intended for small instances (|T| <= 30) where a provably optimal
makespan is wanted to report the optimality gap of the heuristics and the EA.
"""

from dataclasses import dataclass

import networkx as nx
import pulp

from src.core.models import ProblemInstance
from src.solvers.heuristics import solve_hlfet


@dataclass(frozen=True)
class MilpResult:
    """Stores the result of an exact MILP scheduling run.

    Args:
        makespan: Best makespan found, or None if no feasible solution was found.
        is_optimal: True if CBC proved the makespan optimal.
        status: PuLP solver status string (e.g. "Optimal", "Not Solved").
        start_times: Start time per task number, or None if no solution was found.
    """

    makespan: int | None
    is_optimal: bool
    status: str
    start_times: dict[int, int] | None


@dataclass(frozen=True)
class _ModelInputs:
    """Stores the precomputed inputs of the time-indexed MILP model.

    Args:
        graph: Precedence graph.
        task_numbers: Topologically sorted task numbers.
        processing_times: Processing time per task number.
        processor_count: Number of identical processors.
        horizon: Scheduling horizon (a feasible makespan upper bound).
        start_windows: Allowed start times per task number.
    """

    graph: nx.DiGraph
    task_numbers: list[int]
    processing_times: dict[int, int]
    processor_count: int
    horizon: int
    start_windows: dict[int, range]


def solve_milp(
    problem_instance: ProblemInstance,
    time_limit_seconds: float | None = None,
) -> MilpResult:
    """Solves a scheduling instance to optimality with a time-indexed MILP.

    Args:
        problem_instance: Scheduling problem instance with a configured group count.
        time_limit_seconds: Optional CBC wall-clock limit; None runs to completion.

    Returns:
        Optimal (or best incumbent) makespan, optimality flag, status, and schedule.

    Raises:
        ValueError: If the graph is cyclic, task times are invalid, or processors
            are not configured.
    """
    processor_count = problem_instance.number_of_groups
    if processor_count is None or processor_count <= 0:
        raise ValueError("MILP requires a positive number of processors.")

    graph = problem_instance.priority_relationship
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("MILP requires a directed acyclic graph.")

    # solve_hlfet validates the task-time indexes and yields a feasible makespan
    # to use as the (optimal-feasible) scheduling horizon.
    horizon = solve_hlfet(problem_instance).makespan
    task_numbers = list(nx.topological_sort(graph))

    if not task_numbers:
        return MilpResult(makespan=0, is_optimal=True, status="Optimal", start_times={})

    processing_times = {
        task_number: int(problem_instance.task_times[task_number])
        for task_number in task_numbers
    }
    model_inputs = _ModelInputs(
        graph=graph,
        task_numbers=task_numbers,
        processing_times=processing_times,
        processor_count=processor_count,
        horizon=horizon,
        start_windows=_calculate_start_windows(
            graph, task_numbers, processing_times, horizon
        ),
    )

    return _solve_time_indexed_model(model_inputs, time_limit_seconds)


def _solve_time_indexed_model(
    model_inputs: _ModelInputs,
    time_limit_seconds: float | None,
) -> MilpResult:
    """Builds and solves the time-indexed MILP model.

    Args:
        model_inputs: Precomputed model inputs.
        time_limit_seconds: Optional CBC wall-clock limit.

    Returns:
        MILP solution wrapped in a MilpResult.
    """
    problem = pulp.LpProblem("task_scheduling", pulp.LpMinimize)
    start_variables = {
        task_number: {
            start_time: pulp.LpVariable(
                f"x_{task_number}_{start_time}", cat=pulp.LpBinary
            )
            for start_time in model_inputs.start_windows[task_number]
        }
        for task_number in model_inputs.task_numbers
    }
    makespan_variable = pulp.LpVariable(
        "makespan", lowBound=0, upBound=model_inputs.horizon, cat=pulp.LpInteger
    )

    problem += makespan_variable

    for task_number in model_inputs.task_numbers:
        task_starts = start_variables[task_number]
        problem += pulp.lpSum(task_starts.values()) == 1, f"assign_{task_number}"
        start_expression = pulp.lpSum(
            start_time * variable for start_time, variable in task_starts.items()
        )
        problem += (
            makespan_variable
            >= start_expression + model_inputs.processing_times[task_number],
            f"makespan_{task_number}",
        )

    _add_precedence_constraints(problem, model_inputs, start_variables)
    _add_resource_constraints(problem, model_inputs, start_variables)

    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit_seconds)
    problem.solve(solver)
    status = pulp.LpStatus[problem.status]

    return _build_result(status, makespan_variable, start_variables)


def _add_precedence_constraints(
    problem: pulp.LpProblem,
    model_inputs: _ModelInputs,
    start_variables: dict[int, dict[int, pulp.LpVariable]],
) -> None:
    """Adds start-time precedence constraints for every graph edge.

    Args:
        problem: MILP problem being built.
        model_inputs: Precomputed model inputs.
        start_variables: Start indicator variables per task number.
    """
    for predecessor_number, successor_number in model_inputs.graph.edges:
        predecessor_start = pulp.lpSum(
            start_time * variable
            for start_time, variable in start_variables[predecessor_number].items()
        )
        successor_start = pulp.lpSum(
            start_time * variable
            for start_time, variable in start_variables[successor_number].items()
        )
        problem += (
            successor_start
            >= predecessor_start + model_inputs.processing_times[predecessor_number],
            f"precedence_{predecessor_number}_{successor_number}",
        )


def _add_resource_constraints(
    problem: pulp.LpProblem,
    model_inputs: _ModelInputs,
    start_variables: dict[int, dict[int, pulp.LpVariable]],
) -> None:
    """Adds per-time-step processor capacity constraints.

    Args:
        problem: MILP problem being built.
        model_inputs: Precomputed model inputs.
        start_variables: Start indicator variables per task number.
    """
    processing_times = model_inputs.processing_times

    for time_step in range(model_inputs.horizon):
        running_variables = [
            start_variables[task_number][start_time]
            for task_number in model_inputs.task_numbers
            if processing_times[task_number] > 0
            for start_time in start_variables[task_number]
            if start_time <= time_step < start_time + processing_times[task_number]
        ]
        if running_variables:
            problem += (
                pulp.lpSum(running_variables) <= model_inputs.processor_count,
                f"capacity_{time_step}",
            )


def _build_result(
    status: str,
    makespan_variable: pulp.LpVariable,
    start_variables: dict[int, dict[int, pulp.LpVariable]],
) -> MilpResult:
    """Builds a MilpResult from a solved problem.

    Args:
        status: PuLP solver status string.
        makespan_variable: Makespan decision variable.
        start_variables: Start indicator variables per task number.

    Returns:
        Parsed solution, or an empty result if no solution is available.
    """
    makespan_value = pulp.value(makespan_variable)
    if makespan_value is None:
        return MilpResult(
            makespan=None,
            is_optimal=False,
            status=status,
            start_times=None,
        )

    start_times = {
        task_number: next(
            start_time
            for start_time, variable in task_starts.items()
            if (pulp.value(variable) or 0) > 0.5
        )
        for task_number, task_starts in start_variables.items()
    }

    return MilpResult(
        makespan=int(round(makespan_value)),
        is_optimal=status == "Optimal",
        status=status,
        start_times=start_times,
    )


def _calculate_start_windows(
    graph: nx.DiGraph,
    task_numbers: list[int],
    processing_times: dict[int, int],
    horizon: int,
) -> dict[int, range]:
    """Calculates the allowed start-time window for every task.

    Tasks cannot start before all predecessors finish (earliest start) and must
    finish within the horizon (latest start). Pruning the time-indexed variables
    to this window keeps the model small enough for CBC to prove optimality.

    Args:
        graph: Precedence graph.
        task_numbers: Topologically sorted task numbers.
        processing_times: Processing time per task number.
        horizon: Scheduling horizon (a feasible makespan upper bound).

    Returns:
        Allowed start times per task number.
    """
    earliest_start: dict[int, int] = {}
    for task_number in task_numbers:
        earliest_start[task_number] = max(
            (
                earliest_start[predecessor_number]
                + processing_times[predecessor_number]
                for predecessor_number in graph.predecessors(task_number)
            ),
            default=0,
        )

    latest_start: dict[int, int] = {}
    for task_number in reversed(task_numbers):
        latest_finish = min(
            (
                latest_start[successor_number]
                for successor_number in graph.successors(task_number)
            ),
            default=horizon,
        )
        latest_start[task_number] = latest_finish - processing_times[task_number]

    return {
        task_number: range(earliest_start[task_number], latest_start[task_number] + 1)
        for task_number in task_numbers
    }
