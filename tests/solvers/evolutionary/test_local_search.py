"""Tests for CPM local search helpers."""

import networkx as nx
import numpy as np
import pytest

from src.core.decoder import calculate_makespan
from src.core.models import ProblemInstance
from src.solvers.evolutionary import local_search


def test_get_left_boundary_returns_max_predecessor_index() -> None:
    problem_instance = _create_local_search_instance()

    left_boundary = local_search.get_left_boundary(
        problem_instance,
        [0, 1, 2, 3],
        3,
    )

    assert left_boundary == 2


def test_get_left_boundary_returns_negative_one_without_predecessors() -> None:
    problem_instance = _create_local_search_instance()

    left_boundary = local_search.get_left_boundary(
        problem_instance,
        [0, 1, 2, 3],
        0,
    )

    assert left_boundary == -1


def test_create_pulled_left_mutant_moves_task_by_insertion() -> None:
    mutant = local_search.create_pulled_left_mutant([0, 1, 2, 3], 3, 1)

    assert mutant == [0, 3, 1, 2]
    assert sorted(mutant) == [0, 1, 2, 3]


def test_create_pulled_left_mutant_rejects_right_move() -> None:
    with pytest.raises(ValueError, match="left"):
        local_search.create_pulled_left_mutant([0, 1, 2, 3], 1, 2)


def test_cpm_local_search_accepts_equal_or_better_mutant(monkeypatch) -> None:
    problem_instance = _create_local_search_instance()
    initial_permutation = [0, 1, 2, 3]

    monkeypatch.setattr(local_search.random, "choice", lambda _: 1)
    monkeypatch.setattr(local_search.random, "randint", lambda _start, _end: 0)

    improved_permutation = local_search.cpm_local_search(
        problem_instance,
        initial_permutation,
        max_attempts=1,
    )

    assert improved_permutation == [1, 0, 2, 3]
    assert calculate_makespan(improved_permutation, problem_instance) <= (
        calculate_makespan(initial_permutation, problem_instance)
    )


def test_cpm_local_search_returns_original_when_max_attempts_is_zero() -> None:
    problem_instance = _create_local_search_instance()
    initial_permutation = [0, 1, 2, 3]

    final_permutation = local_search.cpm_local_search(
        problem_instance,
        initial_permutation,
        max_attempts=0,
    )

    assert final_permutation == initial_permutation
    assert final_permutation is not initial_permutation


def test_cpm_local_search_ignores_immovable_critical_tasks(monkeypatch) -> None:
    graph = nx.DiGraph()
    graph.add_edge(0, 2)
    graph.add_node(1)
    problem_instance = ProblemInstance(
        number_of_groups=2,
        priority_relationship=graph,
        task_times=np.array([5, 1, 5]),
    )

    def choose_only_movable_task(candidates: list[int]) -> int:
        assert candidates == [2]
        return 2

    monkeypatch.setattr(local_search.random, "choice", choose_only_movable_task)
    monkeypatch.setattr(local_search.random, "randint", lambda _start, _end: 1)

    improved_permutation = local_search.cpm_local_search(
        problem_instance,
        [0, 1, 2],
        max_attempts=1,
    )

    assert improved_permutation == [0, 2, 1]


def test_cpm_local_search_raises_decoder_error_for_invalid_permutation() -> None:
    problem_instance = _create_local_search_instance()

    with pytest.raises(ValueError, match="scheduled before predecessor"):
        local_search.cpm_local_search(
            problem_instance,
            [0, 3, 1, 2],
            max_attempts=1,
        )


def _create_local_search_instance() -> ProblemInstance:
    """Creates a simple local search test instance.

    Returns:
        ProblemInstance with one critical movable task.
    """
    graph = nx.DiGraph()
    graph.add_edges_from([(0, 2), (2, 3)])
    graph.add_node(1)

    return ProblemInstance(
        number_of_groups=2,
        priority_relationship=graph,
        task_times=np.array([1, 5, 1, 0]),
    )
