"""Tests for the baseline-step helpers."""

import networkx as nx
import numpy as np

from src.core.models import ProblemInstance
from src.experiments.run_baselines import _lower_bound


def test_lower_bound_uses_critical_path_when_it_dominates() -> None:
    graph = nx.DiGraph()
    graph.add_edges_from([(0, 1), (1, 2)])
    problem_instance = ProblemInstance(
        number_of_groups=4,
        priority_relationship=graph,
        task_times=np.array([2, 3, 4]),
    )

    # Serial chain: critical path 9 dominates the load bound ceil(9/4)=3.
    assert _lower_bound(problem_instance) == 9


def test_lower_bound_uses_load_when_it_dominates() -> None:
    graph = nx.DiGraph()
    graph.add_nodes_from([0, 1, 2])
    problem_instance = ProblemInstance(
        number_of_groups=2,
        priority_relationship=graph,
        task_times=np.array([3, 3, 3]),
    )

    # Three independent tasks on two processors: load ceil(9/2)=5 dominates the
    # critical path 3.
    assert _lower_bound(problem_instance) == 5
