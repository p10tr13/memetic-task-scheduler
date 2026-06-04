"""Tests for the exact MILP scheduling solver."""

import networkx as nx
import numpy as np
import pytest

from src.core.models import ProblemInstance
from src.solvers.milp import solve_milp


def test_solve_milp_chain_runs_serially() -> None:
    graph = nx.DiGraph()
    graph.add_edges_from([(0, 1), (1, 2)])
    problem_instance = ProblemInstance(
        number_of_groups=2,
        priority_relationship=graph,
        task_times=np.array([2, 3, 4]),
    )

    result = solve_milp(problem_instance)

    assert result.is_optimal
    assert result.makespan == 9


def test_solve_milp_chain_start_times_respect_precedence() -> None:
    graph = nx.DiGraph()
    graph.add_edges_from([(0, 1), (1, 2)])
    problem_instance = ProblemInstance(
        number_of_groups=2,
        priority_relationship=graph,
        task_times=np.array([2, 3, 4]),
    )

    result = solve_milp(problem_instance)

    assert result.start_times == {0: 0, 1: 2, 2: 5}


def test_solve_milp_independent_tasks_pack_onto_processors() -> None:
    graph = nx.DiGraph()
    graph.add_nodes_from([0, 1, 2])
    problem_instance = ProblemInstance(
        number_of_groups=2,
        priority_relationship=graph,
        task_times=np.array([2, 2, 2]),
    )

    result = solve_milp(problem_instance)

    assert result.is_optimal
    assert result.makespan == 4


def test_solve_milp_single_processor_sums_all_times() -> None:
    graph = nx.DiGraph()
    graph.add_nodes_from([0, 1, 2])
    problem_instance = ProblemInstance(
        number_of_groups=1,
        priority_relationship=graph,
        task_times=np.array([2, 3, 4]),
    )

    result = solve_milp(problem_instance)

    assert result.is_optimal
    assert result.makespan == 9


def test_solve_milp_diamond_overlaps_parallel_branch() -> None:
    graph = nx.DiGraph()
    graph.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 3)])
    problem_instance = ProblemInstance(
        number_of_groups=2,
        priority_relationship=graph,
        task_times=np.array([1, 5, 2, 1]),
    )

    result = solve_milp(problem_instance)

    assert result.is_optimal
    assert result.makespan == 7


def test_solve_milp_rejects_missing_processor_count() -> None:
    graph = nx.DiGraph()
    graph.add_nodes_from([0, 1])
    problem_instance = ProblemInstance(
        number_of_groups=None,
        priority_relationship=graph,
        task_times=np.array([1, 2]),
    )

    with pytest.raises(ValueError, match="number of processors"):
        solve_milp(problem_instance)


def test_solve_milp_rejects_cycle() -> None:
    graph = nx.DiGraph()
    graph.add_edges_from([(0, 1), (1, 0)])
    problem_instance = ProblemInstance(
        number_of_groups=1,
        priority_relationship=graph,
        task_times=np.array([1, 1]),
    )

    with pytest.raises(ValueError, match="acyclic"):
        solve_milp(problem_instance)
