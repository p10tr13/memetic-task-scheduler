"""Tests for the Kasahara STG parser."""

from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from src.data.kasahara_parser import parse_stg_file


def test_parse_stg_file_returns_problem_instance(tmp_path: Path) -> None:
    stg_path = tmp_path / "simple.stg"
    stg_path.write_text(
        "\n".join(
            [
                "2",
                "0 0 0",
                "1 5 1 0",
                "2 7 1 1",
                "3 0 1 2",
                "# metadata",
            ]
        ),
        encoding="utf-8",
    )

    problem_instance = parse_stg_file(stg_path)

    assert problem_instance.number_of_groups is None
    assert np.array_equal(problem_instance.task_times, np.array([0, 5, 7, 0]))
    assert set(problem_instance.priority_relationship.nodes) == {0, 1, 2, 3}
    assert set(problem_instance.priority_relationship.edges) == {
        (0, 1),
        (1, 2),
        (2, 3),
    }
    assert nx.is_directed_acyclic_graph(problem_instance.priority_relationship)


def test_parse_stg_file_rejects_cycle(tmp_path: Path) -> None:
    stg_path = tmp_path / "cycle.stg"
    stg_path.write_text(
        "\n".join(
            [
                "2",
                "0 0 0",
                "1 5 1 2",
                "2 7 1 1",
                "3 0 2 1 2",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="cycle"):
        parse_stg_file(stg_path)


def test_parse_stg_file_rejects_invalid_predecessor_count(tmp_path: Path) -> None:
    stg_path = tmp_path / "invalid_predecessor_count.stg"
    stg_path.write_text(
        "\n".join(
            [
                "1",
                "0 0 0",
                "1 5 2 0",
                "2 0 1 1",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="declares 2 predecessors"):
        parse_stg_file(stg_path)


def test_parse_stg_file_rejects_invalid_predecessor_reference(tmp_path: Path) -> None:
    stg_path = tmp_path / "invalid_reference.stg"
    stg_path.write_text(
        "\n".join(
            [
                "1",
                "0 0 0",
                "1 5 1 9",
                "2 0 1 1",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid predecessor"):
        parse_stg_file(stg_path)
