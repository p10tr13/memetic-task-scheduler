"""Benchmark graph registry for the experiment runners.

Defines *which* benchmark instances exist and how to load each one: the bundled
Kasahara STG graphs and the reproducible synthetic graphs (small ones for the
MILP optimality gap, matched sparse/dense pairs for hypothesis H2). Synthetic
graphs use fixed generation seeds so the set is identical on every run.

This module describes the *problem* only — the graph and the processor count it
is scheduled onto. It deliberately holds **no EA hyperparameters**: the search
budget is an experiment knob and lives with documented global defaults in
:mod:`src.experiments.run_single`, not per-instance here.
"""

from dataclasses import dataclass, replace
from pathlib import Path

from src.core.models import ProblemInstance
from src.data.kasahara_parser import parse_stg_file
from src.data.synthetic_generator import generate_synthetic_instance

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPOSITORY_ROOT / "data"

# Real-task count at or below which the time-indexed MILP stays tractable.
MILP_TASK_LIMIT = 30


@dataclass(frozen=True)
class Graph:
    """One benchmark graph: a Kasahara file or synthetic params, on M processors.

    Exactly one of ``file`` or ``synthetic`` is set. ``synthetic`` is the tuple
    ``(task_count, edge_density, generation_seed)``.

    Args:
        processors: Number of processors the graph is scheduled onto.
        file: Path of a Kasahara STG file, relative to ``data/``.
        synthetic: Generator parameters ``(task_count, edge_density, seed)``.
    """

    processors: int
    file: str | None = None
    synthetic: tuple[int, float, int] | None = None


@dataclass(frozen=True)
class InstanceMetadata:
    """Derived size metrics of a benchmark instance.

    Args:
        nodes: Number of graph nodes (including dummy entry/exit nodes).
        edges: Number of precedence edges.
        density: Edge density measured as ``edges / nodes**2`` (the H2 measure).
    """

    nodes: int
    edges: int
    density: float


@dataclass(frozen=True)
class InstanceEntry:
    """A materialized benchmark graph.

    Args:
        name: Unique instance identifier used in result files.
        problem_instance: Scheduling instance with its processor count set.
        supports_milp: Whether the instance is small enough for the exact MILP.
        metadata: Derived size metrics (nodes, edges, density).
    """

    name: str
    problem_instance: ProblemInstance
    supports_milp: bool
    metadata: InstanceMetadata


# Full benchmark set. Synthetic graphs (generated in-memory from fixed seeds;
# `make data` writes the same graphs to data/synthetic/*.stg). Naming:
# syn_t<tasks>_p<round(edge_density*100), 3-digit> / <rand-index>_t<tasks>_p<density>.
GRAPHS = {
    # Synthetic, spanning size and density.
    "syn_t15_p010": Graph(processors=2, synthetic=(15, 0.10, 11)),
    "syn_t22_p018": Graph(processors=3, synthetic=(22, 0.18, 12)),
    "syn_t30_p030": Graph(processors=3, synthetic=(30, 0.30, 13)),
    "syn_t40_p006": Graph(processors=4, synthetic=(40, 0.06, 21)),
    "syn_t40_p035": Graph(processors=4, synthetic=(40, 0.35, 23)),
    "syn_t60_p005": Graph(processors=4, synthetic=(60, 0.05, 31)),
    "syn_t60_p015": Graph(processors=4, synthetic=(60, 0.15, 32)),
    "syn_t100_p004": Graph(processors=4, synthetic=(100, 0.04, 41)),
    "syn_t100_p012": Graph(processors=4, synthetic=(100, 0.12, 42)),
    "syn_t150_p004": Graph(processors=4, synthetic=(150, 0.04, 51)),
    # Bundled Kasahara STG graphs (zero-communication ver2 + application graphs).
    "rand0000_t50_p008": Graph(processors=4, file="kasahara/rand0000_t50_p008.stg"),
    "rand0093_t50_p078": Graph(processors=4, file="kasahara/rand0093_t50_p078.stg"),
    "rand0063_t100_p002": Graph(processors=4, file="kasahara/rand0063_t100_p002.stg"),
    "rand0173_t100_p039": Graph(processors=4, file="kasahara/rand0173_t100_p039.stg"),
    "rand0071_t300_p001": Graph(processors=4, file="kasahara/rand0071_t300_p001.stg"),
    "rand0000_t300_p016": Graph(processors=4, file="kasahara/rand0000_t300_p016.stg"),
    "rand0014_t500_p020": Graph(processors=8, file="kasahara/rand0014_t500_p020.stg"),
    "robot_t88_p003": Graph(processors=4, file="kasahara/robot_t88_p003.stg"),
    "sparse_t96_p001": Graph(processors=4, file="kasahara/sparse_t96_p001.stg"),
    "fpppp_t334_p002": Graph(processors=4, file="kasahara/fpppp_t334_p002.stg"),
}

# Tiny synthetic graphs for `--quick` smoke tests only.
QUICK_GRAPHS = {
    "quick_small": Graph(processors=2, synthetic=(10, 0.2, 1)),
    "quick_sparse": Graph(processors=3, synthetic=(20, 0.05, 2)),
    "quick_dense": Graph(processors=3, synthetic=(20, 0.3, 3)),
}

_INDEX = {**GRAPHS, **QUICK_GRAPHS}


def list_instances(quick: bool = False) -> list[str]:
    """Returns benchmark instance names in deterministic order.

    Args:
        quick: When True, returns the tiny quick-smoke set instead of the full
            campaign set.

    Returns:
        Instance names for the requested set.
    """
    return list(QUICK_GRAPHS if quick else GRAPHS)


def get_instance(name: str) -> InstanceEntry:
    """Materializes one named benchmark graph.

    Args:
        name: Registry instance name (see :func:`list_instances`).

    Returns:
        The materialized graph with processor count, metadata and MILP support.

    Raises:
        KeyError: If the name is not in the registry.
    """
    graph = _INDEX[name]
    if graph.file is not None:
        base_instance = parse_stg_file(DATA_DIR / graph.file)
    else:
        task_count, edge_density, generation_seed = graph.synthetic
        base_instance = generate_synthetic_instance(
            task_count=task_count,
            edge_density=edge_density,
            random_seed=generation_seed,
        )

    problem_instance = replace(base_instance, number_of_groups=graph.processors)
    priority_graph = problem_instance.priority_relationship
    nodes = priority_graph.number_of_nodes()
    edges = priority_graph.number_of_edges()

    return InstanceEntry(
        name=name,
        problem_instance=problem_instance,
        supports_milp=nodes <= MILP_TASK_LIMIT + 2,
        metadata=InstanceMetadata(
            nodes=nodes,
            edges=edges,
            density=edges / nodes**2 if nodes else 0.0,
        ),
    )
