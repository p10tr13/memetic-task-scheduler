"""Tests for Critical Path Method utilities."""

import networkx as nx
import numpy as np
import pytest

from src.core.cpm_utils import identify_critical_tasks
from src.core.models import ProblemInstance


def test_identify_critical_tasks_returns_all_tasks_for_chain() -> None:
    graph = nx.DiGraph()
    graph.add_edges_from([(0, 1), (1, 2)])
    problem_instance = ProblemInstance(
        number_of_groups=1,
        priority_relationship=graph,
        task_times=np.array([2, 3, 4]),
    )

    critical_tasks = identify_critical_tasks([0, 1, 2], problem_instance)

    assert critical_tasks == {0, 1, 2}


def test_identify_critical_tasks_follows_longer_precedence_branch() -> None:
    graph = nx.DiGraph()
    graph.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 3)])
    problem_instance = ProblemInstance(
        number_of_groups=2,
        priority_relationship=graph,
        task_times=np.array([0, 5, 2, 0]),
    )

    critical_tasks = identify_critical_tasks([0, 1, 2, 3], problem_instance)

    # The makespan is driven by task 1 -> task 3; the shorter branch (task 2) and
    # the zero-duration entry node carry slack.
    assert critical_tasks == {1, 3}


def test_identify_critical_tasks_includes_resource_predecessor() -> None:
    graph = nx.DiGraph()
    graph.add_nodes_from([0, 1, 2])
    problem_instance = ProblemInstance(
        number_of_groups=2,
        priority_relationship=graph,
        task_times=np.array([5, 5, 5]),
    )

    critical_tasks = identify_critical_tasks([0, 1, 2], problem_instance)

    # Three independent tasks on two processors: task 2 reuses task 0's
    # processor, so the critical chain is the resource path {0, 2}, not the whole
    # graph that a graph-only critical path would report.
    assert critical_tasks == {0, 2}


def test_identify_critical_tasks_rejects_cycle() -> None:
    graph = nx.DiGraph()
    graph.add_edges_from([(0, 1), (1, 0)])
    problem_instance = ProblemInstance(
        number_of_groups=1,
        priority_relationship=graph,
        task_times=np.array([1, 1]),
    )

    with pytest.raises(ValueError, match="acyclic"):
        identify_critical_tasks([0, 1], problem_instance)
