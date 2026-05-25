"""Parser for Kasahara Lab standard task graph files."""

from pathlib import Path

import networkx as nx
import numpy as np

from src.core.models import ProblemInstance


def parse_stg_file(file_path: str | Path) -> ProblemInstance:
    """Parses a Kasahara STG file into a problem instance.

    Args:
        file_path: Path to a Kasahara `.stg` task graph file.

    Returns:
        ProblemInstance with no configured group count.

    Raises:
        ValueError: If the file content is malformed or does not define a DAG.
    """
    path = Path(file_path)
    graph_lines = _read_graph_lines(path)
    task_count = _parse_task_count(graph_lines[0])
    expected_graph_row_count = task_count + 2

    if len(graph_lines) < expected_graph_row_count + 1:
        raise ValueError("STG file does not contain all task graph rows.")

    priority_relationship = nx.DiGraph()
    task_times = np.zeros(expected_graph_row_count, dtype=int)
    seen_task_numbers: set[int] = set()

    for row_index, line in enumerate(graph_lines[1 : expected_graph_row_count + 1]):
        task_number, processing_time, predecessor_numbers = _parse_task_row(line)

        if task_number in seen_task_numbers:
            raise ValueError(f"Duplicate task number: {task_number}.")
        if task_number != row_index:
            raise ValueError(f"Expected task number {row_index}, got {task_number}.")
        if processing_time < 0:
            raise ValueError(f"Task {task_number} has a negative processing time.")

        seen_task_numbers.add(task_number)
        priority_relationship.add_node(task_number)
        task_times[task_number] = processing_time

        for predecessor_number in predecessor_numbers:
            if predecessor_number < 0 or predecessor_number >= expected_graph_row_count:
                raise ValueError(
                    f"Task {task_number} references invalid predecessor "
                    f"{predecessor_number}."
                )
            priority_relationship.add_edge(predecessor_number, task_number)

    if not nx.is_directed_acyclic_graph(priority_relationship):
        raise ValueError("STG task graph contains a cycle.")

    return ProblemInstance(
        number_of_groups=None,
        priority_relationship=priority_relationship,
        task_times=task_times,
    )


def _read_graph_lines(file_path: Path) -> list[str]:
    """Reads non-empty graph lines before STG metadata comments.

    Args:
        file_path: Path to the STG file.

    Returns:
        Non-empty lines up to the first metadata comment.
    """
    graph_lines = []

    with file_path.open(encoding="utf-8") as stg_file:
        for line in stg_file:
            stripped_line = line.strip()
            if not stripped_line:
                continue
            if stripped_line.startswith("#"):
                break
            graph_lines.append(line)

    if not graph_lines:
        raise ValueError("STG file is empty.")

    return graph_lines


def _parse_task_count(line: str) -> int:
    """Parses the number of real tasks from the first STG line.

    Args:
        line: First STG task graph line.

    Returns:
        Number of real, non-dummy tasks.

    Raises:
        ValueError: If the count is missing or invalid.
    """
    fields = line.split()

    if len(fields) != 1:
        raise ValueError("First STG line must contain only the task count.")

    task_count = int(fields[0])
    if task_count < 0:
        raise ValueError("Task count cannot be negative.")

    return task_count


def _parse_task_row(line: str) -> tuple[int, int, list[int]]:
    """Parses one STG task row.

    Args:
        line: STG task row containing task number, time, and predecessors.

    Returns:
        Task number, processing time, and predecessor task numbers.

    Raises:
        ValueError: If the row has an invalid field count.
    """
    fields = [int(field) for field in line.split()]

    if len(fields) < 3:
        raise ValueError("Task row must contain at least three fields.")

    task_number = fields[0]
    processing_time = fields[1]
    predecessor_count = fields[2]
    predecessor_numbers = fields[3:]

    if predecessor_count < 0:
        raise ValueError(f"Task {task_number} has a negative predecessor count.")
    if predecessor_count != len(predecessor_numbers):
        raise ValueError(
            f"Task {task_number} declares {predecessor_count} predecessors but "
            f"contains {len(predecessor_numbers)}."
        )

    return task_number, processing_time, predecessor_numbers
