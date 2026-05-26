"""Tests for permutation crossover operators."""

import pytest

from src.solvers.evolutionary.crossovers import ox_crossover, pmx_crossover


def test_ox_crossover_preserves_tasks_without_duplicates(monkeypatch) -> None:
    first_parent = [0, 1, 2, 3, 4, 5]
    second_parent = [3, 5, 4, 1, 0, 2]

    monkeypatch.setattr("random.sample", lambda _population, _sample_size: [1, 3])

    first_child, second_child = ox_crossover(first_parent, second_parent)

    assert sorted(first_child) == [0, 1, 2, 3, 4, 5]
    assert sorted(second_child) == [0, 1, 2, 3, 4, 5]
    assert len(set(first_child)) == len(first_child)
    assert len(set(second_child)) == len(second_child)


def test_ox_crossover_keeps_selected_segment(monkeypatch) -> None:
    first_parent = [0, 1, 2, 3, 4, 5]
    second_parent = [3, 5, 4, 1, 0, 2]

    monkeypatch.setattr("random.sample", lambda _population, _sample_size: [1, 3])

    first_child, second_child = ox_crossover(first_parent, second_parent)

    assert first_child[1:4] == [1, 2, 3]
    assert second_child[1:4] == [5, 4, 1]


def test_pmx_crossover_preserves_tasks_without_duplicates(monkeypatch) -> None:
    first_parent = [0, 1, 2, 3, 4, 5]
    second_parent = [3, 5, 4, 1, 0, 2]

    monkeypatch.setattr("random.sample", lambda _population, _sample_size: [1, 3])

    first_child, second_child = pmx_crossover(first_parent, second_parent)

    assert sorted(first_child) == [0, 1, 2, 3, 4, 5]
    assert sorted(second_child) == [0, 1, 2, 3, 4, 5]
    assert len(set(first_child)) == len(first_child)
    assert len(set(second_child)) == len(second_child)


def test_pmx_crossover_keeps_selected_segment(monkeypatch) -> None:
    first_parent = [0, 1, 2, 3, 4, 5]
    second_parent = [3, 5, 4, 1, 0, 2]

    monkeypatch.setattr("random.sample", lambda _population, _sample_size: [1, 3])

    first_child, second_child = pmx_crossover(first_parent, second_parent)

    assert first_child[1:4] == [1, 2, 3]
    assert second_child[1:4] == [5, 4, 1]


def test_crossovers_mutate_parent_lists_in_place(monkeypatch) -> None:
    first_parent = [0, 1, 2, 3, 4, 5]
    second_parent = [3, 5, 4, 1, 0, 2]

    monkeypatch.setattr("random.sample", lambda _population, _sample_size: [1, 3])

    first_child, second_child = ox_crossover(first_parent, second_parent)

    assert first_child is first_parent
    assert second_child is second_parent


def test_crossovers_reject_duplicate_tasks() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        ox_crossover([0, 1, 1], [0, 1, 2])

    with pytest.raises(ValueError, match="duplicate"):
        pmx_crossover([0, 1, 1], [0, 1, 2])


def test_crossovers_reject_different_task_sets() -> None:
    with pytest.raises(ValueError, match="same tasks"):
        ox_crossover([0, 1, 2], [0, 1, 3])

    with pytest.raises(ValueError, match="same tasks"):
        pmx_crossover([0, 1, 2], [0, 1, 3])
