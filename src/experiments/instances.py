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
DATA_DIR = REPOSITORY_ROOT / "data"

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

# Synthetic instances spanning size and density (generated in-memory from these
# fixed-seed specs; `make data` / scripts.build_data writes the same instances to
# data/synthetic/*.stg). Density label refers to the requested edge density passed
# to the generator. Instances with <= MILP_TASK_LIMIT real tasks get an exact MILP
# reference; the rest use the lower bound.
_SYNTHETIC_SPECS = (
    # Naming: syn_t<tasks>_p<round(real edge density * 100), 3-digit>.
    # name              M  pop gen seeds  (task_count, edge_density, seed)
    _Spec("syn_t15_p010", 2, 30, 40, 30, synthetic=_SyntheticParams(15, 0.10, 11)),
    _Spec("syn_t22_p018", 3, 30, 40, 30, synthetic=_SyntheticParams(22, 0.18, 12)),
    _Spec("syn_t30_p030", 3, 30, 40, 30, synthetic=_SyntheticParams(30, 0.30, 13)),
    _Spec("syn_t40_p006", 4, 40, 60, 30, synthetic=_SyntheticParams(40, 0.06, 21)),
    _Spec("syn_t40_p035", 4, 40, 60, 30, synthetic=_SyntheticParams(40, 0.35, 23)),
    _Spec("syn_t60_p005", 4, 40, 60, 30, synthetic=_SyntheticParams(60, 0.05, 31)),
    _Spec("syn_t60_p015", 4, 40, 60, 30, synthetic=_SyntheticParams(60, 0.15, 32)),
    _Spec("syn_t100_p004", 4, 50, 80, 30, synthetic=_SyntheticParams(100, 0.04, 41)),
    _Spec("syn_t100_p012", 4, 50, 80, 30, synthetic=_SyntheticParams(100, 0.12, 42)),
    _Spec("syn_t150_p004", 4, 50, 80, 20, synthetic=_SyntheticParams(150, 0.04, 51)),
)

# Bundled Kasahara STG graphs (zero-communication ver2 + application graphs),
# chosen to span size and density. See data/optima.csv for published
# optimal/best-known makespans. Provenance (size/rand-index) is in each file header.
_KASAHARA_SPECS = (
    # Naming: <rand-index>_t<tasks>_p<round(real edge density * 100), 3-digit>.
    # name                    M  pop gen seeds  file
    _Spec("rand0000_t50_p008", 4, 40, 60, 30, file_name="kasahara/rand0000_t50_p008.stg"),
    _Spec("rand0093_t50_p078", 4, 40, 60, 30, file_name="kasahara/rand0093_t50_p078.stg"),
    _Spec("rand0063_t100_p002", 4, 50, 80, 30, file_name="kasahara/rand0063_t100_p002.stg"),
    _Spec("rand0173_t100_p039", 4, 50, 80, 30, file_name="kasahara/rand0173_t100_p039.stg"),
    _Spec("rand0071_t300_p001", 4, 50, 80, 15, file_name="kasahara/rand0071_t300_p001.stg"),
    _Spec("rand0000_t300_p016", 4, 50, 80, 15, file_name="kasahara/rand0000_t300_p016.stg"),
    _Spec("rand0014_t500_p020", 8, 60, 100, 10, file_name="kasahara/rand0014_t500_p020.stg"),
    # Application graphs (p = real edge density, same basis as the random graphs).
    _Spec("robot_t88_p003", 4, 50, 80, 30, file_name="kasahara/robot_t88_p003.stg"),
    _Spec("sparse_t96_p001", 4, 50, 80, 30, file_name="kasahara/sparse_t96_p001.stg"),
    _Spec("fpppp_t334_p002", 4, 50, 80, 15, file_name="kasahara/fpppp_t334_p002.stg"),
)

_FULL_SPECS = _SYNTHETIC_SPECS + _KASAHARA_SPECS


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
        base_instance = parse_stg_file(DATA_DIR / spec.file_name)
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
