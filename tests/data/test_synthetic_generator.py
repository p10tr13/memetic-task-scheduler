"""Tests for synthetic task graph generation."""

from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from src.data.kasahara_parser import parse_stg_file
from src.data.synthetic_generator import (
    generate_synthetic_instance,
    write_stg_file,
)


def test_generate_synthetic_instance_is_reproducible_with_seed() -> None:
    first_instance = generate_synthetic_instance(
        task_count=8,
        edge_density=0.3,
        random_seed=123,
    )
    second_instance = generate_synthetic_instance(
        task_count=8,
        edge_density=0.3,
        random_seed=123,
    )

    assert np.array_equal(first_instance.task_times, second_instance.task_times)
    assert set(first_instance.priority_relationship.edges) == set(
        second_instance.priority_relationship.edges
    )


def test_generate_synthetic_instance_adds_dummies_without_density_count() -> None:
    problem_instance = generate_synthetic_instance(
        task_count=4,
        edge_density=0.5,
        processing_time_range=(2, 4),
        random_seed=7,
        add_dummies=True,
    )
    graph = problem_instance.priority_relationship
    real_task_numbers = set(range(1, 5))
    real_edges = [
        edge
        for edge in graph.edges
        if edge[0] in real_task_numbers and edge[1] in real_task_numbers
    ]

    assert set(graph.nodes) == {0, 1, 2, 3, 4, 5}
    assert problem_instance.task_times[0] == 0
    assert problem_instance.task_times[5] == 0
    assert len(real_edges) == 3
    assert nx.is_directed_acyclic_graph(graph)


def test_generate_synthetic_instance_can_skip_dummies() -> None:
    problem_instance = generate_synthetic_instance(
        task_count=4,
        edge_density=0.5,
        random_seed=7,
        add_dummies=False,
    )

    assert set(problem_instance.priority_relationship.nodes) == {0, 1, 2, 3}
    assert len(problem_instance.task_times) == 4
    assert all(processing_time > 0 for processing_time in problem_instance.task_times)


def test_write_stg_file_round_trips_generated_instance(tmp_path: Path) -> None:
    stg_path = tmp_path / "synthetic.stg"
    problem_instance = generate_synthetic_instance(
        task_count=5,
        edge_density=0.4,
        random_seed=42,
        add_dummies=True,
    )

    write_stg_file(
        problem_instance,
        stg_path,
        graph_name="synthetic.stg",
        random_seed=42,
        requested_edge_density=0.4,
    )
    parsed_instance = parse_stg_file(stg_path)

    assert np.array_equal(parsed_instance.task_times, problem_instance.task_times)
    assert set(parsed_instance.priority_relationship.edges) == set(
        problem_instance.priority_relationship.edges
    )


def test_write_stg_file_wraps_instance_without_dummies(tmp_path: Path) -> None:
    stg_path = tmp_path / "synthetic_without_dummies.stg"
    problem_instance = generate_synthetic_instance(
        task_count=3,
        edge_density=1.0,
        random_seed=5,
        add_dummies=False,
    )

    write_stg_file(problem_instance, stg_path)
    parsed_instance = parse_stg_file(stg_path)

    assert set(parsed_instance.priority_relationship.nodes) == {0, 1, 2, 3, 4}
    assert parsed_instance.task_times[0] == 0
    assert parsed_instance.task_times[4] == 0
    assert np.array_equal(
        parsed_instance.task_times[1:4],
        problem_instance.task_times,
    )


def test_write_stg_file_uses_eleven_character_numeric_fields(tmp_path: Path) -> None:
    stg_path = tmp_path / "synthetic.stg"
    problem_instance = generate_synthetic_instance(
        task_count=2,
        edge_density=1.0,
        random_seed=10,
    )

    write_stg_file(problem_instance, stg_path)
    graph_lines = [
        line
        for line in stg_path.read_text(encoding="utf-8").splitlines()
        if not line.startswith("#")
    ]

    assert len(graph_lines[0]) == 11
    assert all(len(line) % 11 == 0 for line in graph_lines)


def test_generate_synthetic_instance_rejects_invalid_density() -> None:
    with pytest.raises(ValueError, match="Edge density"):
        generate_synthetic_instance(
            task_count=4,
            edge_density=1.5,
            random_seed=1,
        )
