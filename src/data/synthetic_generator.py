"""Synthetic DAG generation and STG export utilities."""

import random
from pathlib import Path
from typing import NamedTuple

import networkx as nx
import numpy as np

from src.core.models import ProblemInstance

FIELD_WIDTH = 11


def generate_synthetic_instance(
    task_count: int,
    edge_density: float,
    processing_time_range: tuple[int, int] = (1, 10),
    random_seed: int | None = None,
    add_dummies: bool = True,
) -> ProblemInstance:
    """Generates a reproducible synthetic DAG scheduling instance.

    Args:
        task_count: Number of non-dummy tasks to generate.
        edge_density: Fraction of possible non-dummy precedence edges to include.
        processing_time_range: Inclusive minimum and maximum processing time.
        random_seed: Seed used by the local random number generator.
        add_dummies: Whether to add entry and exit dummy nodes with zero duration.

    Returns:
        Synthetic problem instance.

    Raises:
        ValueError: If generation parameters are invalid.
    """
    min_processing_time, max_processing_time = processing_time_range
    _validate_generation_parameters(
        task_count,
        edge_density,
        min_processing_time,
        max_processing_time,
    )

    random_generator = random.Random(random_seed)
    real_task_numbers = _create_real_task_numbers(task_count, add_dummies)
    graph = nx.DiGraph()
    graph.add_nodes_from(real_task_numbers)
    task_times = np.zeros(task_count + 2 if add_dummies else task_count, dtype=int)

    for task_number in real_task_numbers:
        task_times[task_number] = random_generator.randint(
            min_processing_time,
            max_processing_time,
        )

    possible_edges = [
        (predecessor_number, successor_number)
        for predecessor_index, predecessor_number in enumerate(real_task_numbers)
        for successor_number in real_task_numbers[predecessor_index + 1 :]
    ]
    selected_edge_count = round(edge_density * len(possible_edges))
    graph.add_edges_from(random_generator.sample(possible_edges, selected_edge_count))

    if add_dummies:
        _add_dummy_nodes(graph, task_times, real_task_numbers)

    return ProblemInstance(
        number_of_groups=None,
        priority_relationship=graph,
        task_times=task_times,
    )


def write_stg_file(
    problem_instance: ProblemInstance,
    file_path: str | Path,
    graph_name: str = "synthetic.stg",
    random_seed: int | None = None,
    requested_edge_density: float | None = None,
) -> None:
    """Writes a problem instance to a Kasahara-compatible STG file.

    Args:
        problem_instance: Problem instance to serialize.
        file_path: Destination STG file path.
        graph_name: Name recorded in the metadata section.
        random_seed: Optional seed recorded in the metadata section.
        requested_edge_density: Optional requested density recorded in metadata.

    Raises:
        ValueError: If the instance graph is not a DAG or has unsupported task IDs.
    """
    if not nx.is_directed_acyclic_graph(problem_instance.priority_relationship):
        raise ValueError("STG export requires a directed acyclic graph.")

    stg_graph_data = _build_stg_graph_data(problem_instance)
    metadata_lines = _create_metadata_lines(
        stg_graph_data,
        graph_name,
        random_seed,
        requested_edge_density,
    )
    output_path = Path(file_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(
            [
                _format_stg_number(stg_graph_data.real_task_count),
                *(
                    _format_stg_row(task_number, processing_time, predecessors)
                    for task_number, processing_time, predecessors in stg_graph_data.rows
                ),
                *metadata_lines,
            ]
        )
        + "\n",
        encoding="utf-8",
    )


class _StgGraphData(NamedTuple):
    """Stores normalized graph data for STG export.

    Args:
        real_task_count: Number of non-dummy tasks in the STG file.
        rows: STG rows containing task number, processing time, and predecessors.
        real_edge_count: Number of edges between non-dummy tasks.
        dummy_edge_count: Number of edges connected to dummy tasks.
        critical_path_length: Longest weighted path length.
        total_processing_time: Sum of non-dummy processing times.
    """

    real_task_count: int
    rows: list[tuple[int, int, list[int]]]
    real_edge_count: int
    dummy_edge_count: int
    critical_path_length: int
    total_processing_time: int


def _validate_generation_parameters(
    task_count: int,
    edge_density: float,
    min_processing_time: int,
    max_processing_time: int,
) -> None:
    """Validates synthetic instance generation parameters.

    Args:
        task_count: Number of non-dummy tasks.
        edge_density: Requested non-dummy edge density.
        min_processing_time: Minimum non-dummy processing time.
        max_processing_time: Maximum non-dummy processing time.

    Raises:
        ValueError: If any parameter is invalid.
    """
    if task_count < 0:
        raise ValueError("Task count cannot be negative.")
    if not 0.0 <= edge_density <= 1.0:
        raise ValueError("Edge density must be between 0 and 1.")
    if min_processing_time <= 0:
        raise ValueError("Minimum processing time must be positive.")
    if max_processing_time < min_processing_time:
        raise ValueError("Maximum processing time cannot be below minimum.")


def _create_real_task_numbers(task_count: int, add_dummies: bool) -> list[int]:
    """Creates task IDs for generated non-dummy tasks.

    Args:
        task_count: Number of non-dummy tasks.
        add_dummies: Whether dummy task IDs are reserved.

    Returns:
        Non-dummy task IDs.
    """
    if add_dummies:
        return list(range(1, task_count + 1))

    return list(range(task_count))


def _add_dummy_nodes(
    graph: nx.DiGraph,
    task_times: np.ndarray,
    real_task_numbers: list[int],
) -> None:
    """Adds entry and exit dummy nodes to a generated graph.

    Args:
        graph: Graph containing non-dummy tasks.
        task_times: Processing times indexed by task number.
        real_task_numbers: Non-dummy task IDs.
    """
    entry_dummy_number = 0
    exit_dummy_number = len(task_times) - 1
    graph.add_nodes_from([entry_dummy_number, exit_dummy_number])

    for task_number in real_task_numbers:
        if graph.in_degree(task_number) == 0:
            graph.add_edge(entry_dummy_number, task_number)
        if graph.out_degree(task_number) == 0:
            graph.add_edge(task_number, exit_dummy_number)


def _build_stg_graph_data(problem_instance: ProblemInstance) -> _StgGraphData:
    """Builds normalized Kasahara STG rows from a problem instance.

    Args:
        problem_instance: Problem instance to normalize.

    Returns:
        Normalized STG graph data.

    Raises:
        ValueError: If task IDs are unsupported.
    """
    graph = problem_instance.priority_relationship
    task_numbers = sorted(graph.nodes)

    if task_numbers != list(range(len(task_numbers))):
        raise ValueError("STG export requires contiguous task numbers starting at 0.")

    if _has_stg_dummy_nodes(problem_instance):
        return _build_existing_dummy_stg_data(problem_instance)

    return _build_wrapped_stg_data(problem_instance)


def _has_stg_dummy_nodes(problem_instance: ProblemInstance) -> bool:
    """Checks whether an instance already uses Kasahara dummy nodes.

    Args:
        problem_instance: Problem instance to inspect.

    Returns:
        True if entry and exit dummy nodes are already present.
    """
    graph = problem_instance.priority_relationship
    task_numbers = sorted(graph.nodes)

    if len(task_numbers) < 2:
        return False

    entry_dummy_number = task_numbers[0]
    exit_dummy_number = task_numbers[-1]

    return (
        entry_dummy_number == 0
        and int(problem_instance.task_times[entry_dummy_number]) == 0
        and int(problem_instance.task_times[exit_dummy_number]) == 0
        and graph.in_degree(entry_dummy_number) == 0
        and graph.out_degree(exit_dummy_number) == 0
    )


def _build_existing_dummy_stg_data(
    problem_instance: ProblemInstance,
) -> _StgGraphData:
    """Builds STG data from an instance that already has dummy nodes.

    Args:
        problem_instance: Problem instance with Kasahara dummy nodes.

    Returns:
        Normalized STG graph data.
    """
    graph = problem_instance.priority_relationship
    exit_dummy_number = max(graph.nodes)
    real_task_numbers = list(range(1, exit_dummy_number))
    rows = [
        (
            task_number,
            int(problem_instance.task_times[task_number]),
            sorted(graph.predecessors(task_number)),
        )
        for task_number in range(exit_dummy_number + 1)
    ]
    real_edges = [
        edge
        for edge in graph.edges
        if edge[0] in real_task_numbers and edge[1] in real_task_numbers
    ]
    dummy_edges = [edge for edge in graph.edges if edge not in real_edges]

    return _StgGraphData(
        real_task_count=len(real_task_numbers),
        rows=rows,
        real_edge_count=len(real_edges),
        dummy_edge_count=len(dummy_edges),
        critical_path_length=_calculate_critical_path_length(graph, problem_instance),
        total_processing_time=sum(
            int(problem_instance.task_times[task_number])
            for task_number in real_task_numbers
        ),
    )


def _build_wrapped_stg_data(problem_instance: ProblemInstance) -> _StgGraphData:
    """Builds STG data by wrapping an instance in dummy nodes.

    Args:
        problem_instance: Problem instance without Kasahara dummy nodes.

    Returns:
        Normalized STG graph data.
    """
    graph = problem_instance.priority_relationship
    real_task_count = len(graph.nodes)
    exit_dummy_number = real_task_count + 1
    mapped_graph = nx.DiGraph()
    mapped_graph.add_nodes_from(range(exit_dummy_number + 1))
    task_times = np.zeros(exit_dummy_number + 1, dtype=int)

    for task_number in graph.nodes:
        mapped_task_number = task_number + 1
        task_times[mapped_task_number] = int(problem_instance.task_times[task_number])

    for predecessor_number, successor_number in graph.edges:
        mapped_graph.add_edge(predecessor_number + 1, successor_number + 1)

    real_task_numbers = list(range(1, real_task_count + 1))
    _add_dummy_nodes(mapped_graph, task_times, real_task_numbers)
    wrapped_instance = ProblemInstance(
        number_of_groups=problem_instance.number_of_groups,
        priority_relationship=mapped_graph,
        task_times=task_times,
    )

    return _build_existing_dummy_stg_data(wrapped_instance)


def _calculate_critical_path_length(
    graph: nx.DiGraph,
    problem_instance: ProblemInstance,
) -> int:
    """Calculates the longest weighted path length in a DAG.

    Args:
        graph: Directed acyclic graph.
        problem_instance: Problem instance with processing times.

    Returns:
        Critical path length.
    """
    longest_finish_times: dict[int, int] = {}

    for task_number in nx.topological_sort(graph):
        longest_start_time = max(
            (
                longest_finish_times[predecessor_number]
                for predecessor_number in graph.predecessors(task_number)
            ),
            default=0,
        )
        longest_finish_times[task_number] = longest_start_time + int(
            problem_instance.task_times[task_number]
        )

    return max(longest_finish_times.values(), default=0)


def _create_metadata_lines(
    stg_graph_data: _StgGraphData,
    graph_name: str,
    random_seed: int | None,
    requested_edge_density: float | None,
) -> list[str]:
    """Creates a minimal Kasahara-style metadata section.

    Args:
        stg_graph_data: Normalized STG graph data.
        graph_name: Name recorded in metadata.
        random_seed: Optional generation seed.
        requested_edge_density: Optional requested edge density.

    Returns:
        Metadata lines.
    """
    possible_edge_count = (
        stg_graph_data.real_task_count * (stg_graph_data.real_task_count - 1) / 2
    )
    real_density = (
        stg_graph_data.real_edge_count / possible_edge_count
        if possible_edge_count
        else 0.0
    )
    requested_density = (
        requested_edge_density if requested_edge_density is not None else real_density
    )
    average_processing_time = (
        stg_graph_data.total_processing_time / stg_graph_data.real_task_count
        if stg_graph_data.real_task_count
        else 0.0
    )
    parallelism = (
        stg_graph_data.total_processing_time / stg_graph_data.critical_path_length
        if stg_graph_data.critical_path_length
        else 0.0
    )

    return [
        "# Standard Task Graph Set Project",
        f"# Random Task Graph synthetic/{graph_name}",
        "# Precedence constraints generator : sameprob",
        f"#   Random Seed       : {_format_optional_seed(random_seed)}",
        f"#   Tasks             : {stg_graph_data.real_task_count} (+dummy tasks : 2)",
        (
            f"#   Edges             : {stg_graph_data.real_edge_count} / "
            f"{int(possible_edge_count)} (+dummy edges : {stg_graph_data.dummy_edge_count})"
        ),
        (
            f"#   Probability       : {requested_density:.6f} "
            f"(Real : {real_density:.6f})"
        ),
        "# Task processing time generator : unifproc",
        f"#   Random Seed       : {_format_optional_seed(random_seed)}",
        f"#   Ave. Proc. Time   : {average_processing_time:.6f}",
        f"# CP Length           : {stg_graph_data.critical_path_length}",
        f"# Parallelism         : {parallelism:.6f}",
        "# Source: Synthetic instance generated by memetic-task-scheduler.",
    ]


def _format_optional_seed(random_seed: int | None) -> str:
    """Formats an optional random seed for metadata.

    Args:
        random_seed: Optional random seed.

    Returns:
        Seed text for metadata.
    """
    if random_seed is None:
        return "unknown"

    return str(random_seed)


def _format_stg_row(
    task_number: int,
    processing_time: int,
    predecessor_numbers: list[int],
) -> str:
    """Formats one STG graph row.

    Args:
        task_number: Task number.
        processing_time: Task processing time.
        predecessor_numbers: Predecessor task numbers.

    Returns:
        STG row with 11-character-wide numeric fields.
    """
    fields = [task_number, processing_time, len(predecessor_numbers)]
    fields.extend(predecessor_numbers)

    return "".join(_format_stg_number(field) for field in fields)


def _format_stg_number(value: int) -> str:
    """Formats a number as an 11-character-wide STG field.

    Args:
        value: Integer value to format.

    Returns:
        Right-aligned STG field.
    """
    return f"{value:{FIELD_WIDTH}d}"
