"""Tests for baseline scheduling heuristics."""

import networkx as nx
import numpy as np
import pytest

from src.core.decoder import calculate_makespan
from src.core.models import ProblemInstance
from src.solvers.heuristics import create_hlfet_permutation, solve_hlfet


def test_create_hlfet_permutation_prioritizes_highest_level_task() -> None:
    graph = nx.DiGraph()
    graph.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 3)])
    problem_instance = ProblemInstance(
        number_of_groups=2,
        priority_relationship=graph,
        task_times=np.array([0, 5, 2, 0]),
    )

    task_permutation = create_hlfet_permutation(problem_instance)

    assert task_permutation == [0, 1, 2, 3]


def test_create_hlfet_permutation_breaks_level_ties_by_task_number() -> None:
    graph = nx.DiGraph()
    graph.add_nodes_from([0, 1, 2])
    problem_instance = ProblemInstance(
        number_of_groups=2,
        priority_relationship=graph,
        task_times=np.array([3, 3, 1]),
    )

    task_permutation = create_hlfet_permutation(problem_instance)

    assert task_permutation == [0, 1, 2]


def test_solve_hlfet_returns_decoded_makespan() -> None:
    graph = nx.DiGraph()
    graph.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 3)])
    problem_instance = ProblemInstance(
        number_of_groups=2,
        priority_relationship=graph,
        task_times=np.array([0, 5, 2, 0]),
    )

    result = solve_hlfet(problem_instance)

    assert result.task_permutation == [0, 1, 2, 3]
    assert result.makespan == calculate_makespan(
        result.task_permutation,
        problem_instance,
    )
    assert result.makespan == 5


def test_solve_hlfet_rejects_cycle() -> None:
    graph = nx.DiGraph()
    graph.add_edges_from([(0, 1), (1, 0)])
    problem_instance = ProblemInstance(
        number_of_groups=1,
        priority_relationship=graph,
        task_times=np.array([1, 1]),
    )

    with pytest.raises(ValueError, match="acyclic"):
        solve_hlfet(problem_instance)


def test_solve_hlfet_rejects_missing_task_time() -> None:
    graph = nx.DiGraph()
    graph.add_nodes_from([0, 1, 2])
    problem_instance = ProblemInstance(
        number_of_groups=1,
        priority_relationship=graph,
        task_times=np.array([1, 2]),
    )

    with pytest.raises(ValueError, match="Task times"):
        solve_hlfet(problem_instance)


def test_solve_hlfet_rejects_missing_processor_count() -> None:
    graph = nx.DiGraph()
    graph.add_nodes_from([0, 1])
    problem_instance = ProblemInstance(
        number_of_groups=None,
        priority_relationship=graph,
        task_times=np.array([1, 2]),
    )

    with pytest.raises(ValueError, match="number of processors"):
        solve_hlfet(problem_instance)
