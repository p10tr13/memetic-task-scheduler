"""Benchmark instance set for the experiment campaign.

Defines the instances used by ``run_experiments`` as code: the bundled Kasahara
STG graphs plus reproducible synthetic instances (small ones for the MILP
optimality gap, matched sparse/dense pairs for hypothesis H2). Synthetic
instances use fixed seeds so the set is identical on every run.
"""

from dataclasses import dataclass, replace
from pathlib import Path

from src.core.models import ProblemInstance
from src.data.kasahara_parser import parse_stg_file
from src.data.synthetic_generator import generate_synthetic_instance
from src.solvers.evolutionary.config import EvolutionConfig

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_DIR = REPOSITORY_ROOT / "data" / "sample"

# Real-task count at or below which the time-indexed MILP stays tractable.
MILP_TASK_LIMIT = 30


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
class ExperimentInstance:
    """A benchmark instance together with its run configuration.

    Args:
        name: Unique instance identifier used in result files.
        problem_instance: Scheduling instance with its processor count set.
        evolution_config: Base EA configuration (population, generations, rates);
            the per-seed random seed is filled in by the runner.
        seed_count: Number of independent seeds to run for this instance.
        supports_milp: Whether the instance is small enough for the exact MILP.
        metadata: Derived size metrics (nodes, edges, density).
    """

    name: str
    problem_instance: ProblemInstance
    evolution_config: EvolutionConfig
    seed_count: int
    supports_milp: bool
    metadata: InstanceMetadata


@dataclass(frozen=True)
class _SyntheticParams:
    """Generation parameters for a synthetic instance."""

    task_count: int
    edge_density: float
    generation_seed: int


@dataclass(frozen=True)
class _Spec:
    """Declarative description of one benchmark instance.

    Exactly one of ``file_name`` (Kasahara STG) or ``synthetic`` is set.
    """

    name: str
    processors: int
    population: int
    generations: int
    seed_count: int
    file_name: str | None = None
    synthetic: _SyntheticParams | None = None


_QUICK_SPECS = (
    _Spec("quick_small", 2, 20, 8, 3, synthetic=_SyntheticParams(10, 0.2, 1)),
    _Spec("quick_sparse", 3, 20, 8, 3, synthetic=_SyntheticParams(20, 0.05, 2)),
    _Spec("quick_dense", 3, 20, 8, 3, synthetic=_SyntheticParams(20, 0.3, 3)),
)

_FULL_SPECS = (
    # Small synthetic instances — exact MILP optimum available.
    _Spec("small_15", 2, 30, 40, 30, synthetic=_SyntheticParams(15, 0.18, 11)),
    _Spec("small_22", 3, 30, 40, 30, synthetic=_SyntheticParams(22, 0.15, 12)),
    # H2 sparse/dense synthetic pairs (n=40).
    _Spec("h2_sparse_a", 4, 40, 60, 30, synthetic=_SyntheticParams(40, 0.06, 21)),
    _Spec("h2_sparse_b", 4, 40, 60, 30, synthetic=_SyntheticParams(40, 0.06, 22)),
    _Spec("h2_dense_a", 4, 40, 60, 30, synthetic=_SyntheticParams(40, 0.35, 23)),
    _Spec("h2_dense_b", 4, 40, 60, 30, synthetic=_SyntheticParams(40, 0.35, 24)),
    # Bundled Kasahara graphs.
    _Spec("example", 3, 40, 60, 30, file_name="example.stg"),
    _Spec("random_50", 4, 40, 60, 30, file_name="random_50.stg"),
    _Spec("robot_control", 4, 50, 100, 10, file_name="robot_control.stg"),
    _Spec("random_300", 4, 30, 40, 10, file_name="random_300.stg"),
)


def build_instances(quick: bool = False) -> list[ExperimentInstance]:
    """Builds the benchmark instance set.

    Args:
        quick: When True, returns a tiny subset with reduced budgets for
            smoke-testing the runner instead of the full campaign set.

    Returns:
        Experiment instances in a deterministic order.
    """
    specs = _QUICK_SPECS if quick else _FULL_SPECS
    return [_materialize(spec) for spec in specs]


def _materialize(spec: _Spec) -> ExperimentInstance:
    """Builds an ExperimentInstance from its declarative spec."""
    if spec.file_name is not None:
        base_instance = parse_stg_file(SAMPLE_DIR / spec.file_name)
    else:
        synthetic = spec.synthetic
        base_instance = generate_synthetic_instance(
            task_count=synthetic.task_count,
            edge_density=synthetic.edge_density,
            random_seed=synthetic.generation_seed,
        )

    problem_instance = replace(base_instance, number_of_groups=spec.processors)
    graph = problem_instance.priority_relationship
    nodes = graph.number_of_nodes()
    edges = graph.number_of_edges()

    return ExperimentInstance(
        name=spec.name,
        problem_instance=problem_instance,
        evolution_config=EvolutionConfig(
            population_size=spec.population,
            generations=spec.generations,
            crossover_probability=0.8,
            mutation_probability=0.3,
        ),
        seed_count=spec.seed_count,
        supports_milp=nodes <= MILP_TASK_LIMIT + 2,
        metadata=InstanceMetadata(
            nodes=nodes,
            edges=edges,
            density=edges / nodes**2 if nodes else 0.0,
        ),
    )
