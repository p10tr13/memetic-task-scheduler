"""Tests for the list scheduling decoder."""

import networkx as nx
import numpy as np
import pytest

from src.core.cpm_utils import identify_critical_tasks
from src.core.decoder import (
    calculate_makespan,
    compute_schedule,
    get_eval_count,
    reset_eval_counter,
    start_eval_counter,
)
from src.core.models import ProblemInstance


def test_calculate_makespan_for_precedence_valid_permutation() -> None:
    problem_instance = _create_problem_instance(number_of_groups=2)

    makespan = calculate_makespan([0, 1, 2, 3], problem_instance)

    assert makespan == 4


def test_calculate_makespan_waits_for_predecessors_and_processors() -> None:
    graph = nx.DiGraph()
    graph.add_edges_from([(0, 2), (1, 2), (2, 3)])
    problem_instance = ProblemInstance(
        number_of_groups=2,
        priority_relationship=graph,
        task_times=np.array([4, 6, 3, 2]),
    )

    makespan = calculate_makespan([0, 1, 2, 3], problem_instance)

    assert makespan == 11


def test_calculate_makespan_rejects_missing_processor_count() -> None:
    problem_instance = _create_problem_instance(number_of_groups=None)

    with pytest.raises(ValueError, match="number of processors"):
        calculate_makespan([0, 1, 2, 3], problem_instance)


def test_calculate_makespan_rejects_non_positive_processor_count() -> None:
    problem_instance = _create_problem_instance(number_of_groups=0)

    with pytest.raises(ValueError, match="positive"):
        calculate_makespan([0, 1, 2, 3], problem_instance)


def test_calculate_makespan_rejects_duplicate_task() -> None:
    problem_instance = _create_problem_instance(number_of_groups=2)

    with pytest.raises(ValueError, match="duplicate"):
        calculate_makespan([0, 1, 1, 3], problem_instance)


def test_calculate_makespan_rejects_missing_task() -> None:
    problem_instance = _create_problem_instance(number_of_groups=2)

    with pytest.raises(ValueError, match="Missing"):
        calculate_makespan([0, 1, 2], problem_instance)


def test_calculate_makespan_rejects_unknown_task() -> None:
    problem_instance = _create_problem_instance(number_of_groups=2)

    with pytest.raises(ValueError, match="Unknown"):
        calculate_makespan([0, 1, 2, 9], problem_instance)


def test_calculate_makespan_rejects_precedence_violation() -> None:
    problem_instance = _create_problem_instance(number_of_groups=2)

    with pytest.raises(ValueError, match="scheduled before predecessor"):
        calculate_makespan([0, 1, 3, 2], problem_instance)


def test_calculate_makespan_rejects_missing_task_time() -> None:
    graph = nx.DiGraph()
    graph.add_nodes_from([0, 1, 2])
    problem_instance = ProblemInstance(
        number_of_groups=1,
        priority_relationship=graph,
        task_times=np.array([1, 2]),
    )

    with pytest.raises(ValueError, match="Task times"):
        calculate_makespan([0, 1, 2], problem_instance)


def test_compute_schedule_returns_start_and_finish_times() -> None:
    problem_instance = _create_problem_instance(number_of_groups=2)

    schedule = compute_schedule([0, 1, 2, 3], problem_instance)

    assert schedule.makespan == 4
    assert schedule.start_times == {0: 0, 1: 0, 2: 0, 3: 4}
    assert schedule.finish_times == {0: 0, 1: 4, 2: 3, 3: 4}
    assert schedule.machine_predecessors == {0: None, 1: 0, 2: None, 3: 2}


def test_compute_schedule_records_processor_reuse() -> None:
    graph = nx.DiGraph()
    graph.add_nodes_from([0, 1, 2])
    problem_instance = ProblemInstance(
        number_of_groups=2,
        priority_relationship=graph,
        task_times=np.array([5, 5, 5]),
    )

    schedule = compute_schedule([0, 1, 2], problem_instance)

    assert schedule.makespan == 10
    assert schedule.machine_predecessors == {0: None, 1: None, 2: 0}


def test_eval_counter_is_inactive_by_default() -> None:
    problem_instance = _create_problem_instance(number_of_groups=2)

    calculate_makespan([0, 1, 2, 3], problem_instance)

    assert get_eval_count() == 0


def test_eval_counter_counts_each_decode() -> None:
    problem_instance = _create_problem_instance(number_of_groups=2)

    token = start_eval_counter()
    try:
        calculate_makespan([0, 1, 2, 3], problem_instance)
        compute_schedule([0, 1, 2, 3], problem_instance)
        assert get_eval_count() == 2
    finally:
        reset_eval_counter(token)

    assert get_eval_count() == 0


def test_eval_counter_counts_critical_path_decodes() -> None:
    problem_instance = _create_problem_instance(number_of_groups=2)

    token = start_eval_counter()
    try:
        identify_critical_tasks([0, 1, 2, 3], problem_instance)
        # ``identify_critical_tasks`` decodes the schedule once via
        # ``compute_schedule``; that decode must be counted too.
        assert get_eval_count() == 1
    finally:
        reset_eval_counter(token)


def test_eval_counter_reset_restores_previous_state() -> None:
    problem_instance = _create_problem_instance(number_of_groups=2)

    outer = start_eval_counter()
    calculate_makespan([0, 1, 2, 3], problem_instance)

    inner = start_eval_counter()
    calculate_makespan([0, 1, 2, 3], problem_instance)
    assert get_eval_count() == 1
    reset_eval_counter(inner)

    assert get_eval_count() == 1
    reset_eval_counter(outer)


def _create_problem_instance(number_of_groups: int | None) -> ProblemInstance:
    """Creates a small scheduling instance for decoder tests.

    Args:
        number_of_groups: Number of processors assigned to the instance.

    Returns:
        ProblemInstance with four tasks and precedence constraints.
    """
    graph = nx.DiGraph()
    graph.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 3)])

    return ProblemInstance(
        number_of_groups=number_of_groups,
        priority_relationship=graph,
        task_times=np.array([0, 4, 3, 0]),
    )
