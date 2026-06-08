"""Materialize the synthetic benchmark instances as ``.stg`` files.

The experiment campaign generates its synthetic instances in memory from the
fixed-seed specs in :mod:`src.experiments.instances`. This script writes those
exact instances out as Kasahara-format ``.stg`` files in ``data/synthetic/`` so
the benchmark is inspectable on disk and reproducible byte-for-byte.

The real Kasahara graphs in ``data/kasahara/`` are downloads and are NOT touched
by this script. The bundled fixtures ``data/synthetic/{example,small}.stg`` are
also left as-is. Run from the repository root::

    python -m scripts.build_data        # or: make data
"""

from pathlib import Path

from src.data.synthetic_generator import generate_synthetic_instance, write_stg_file
from src.experiments.instances import GRAPHS

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_DIR = REPOSITORY_ROOT / "data" / "synthetic"


def _write_graph(name: str, synthetic: tuple, destination_dir: Path) -> Path:
    """Generates one synthetic graph and writes it as an STG file."""
    task_count, edge_density, generation_seed = synthetic
    instance = generate_synthetic_instance(
        task_count=task_count,
        edge_density=edge_density,
        random_seed=generation_seed,
    )
    destination_dir.mkdir(parents=True, exist_ok=True)
    output_path = destination_dir / f"{name}.stg"
    write_stg_file(
        instance,
        output_path,
        graph_name=f"{name}.stg",
        random_seed=generation_seed,
        requested_edge_density=edge_density,
    )
    return output_path


def main() -> None:
    """Writes the synthetic campaign instances to data/synthetic/."""
    written = [
        _write_graph(name, graph.synthetic, SYNTHETIC_DIR)
        for name, graph in GRAPHS.items()
        if graph.synthetic is not None
    ]

    for path in written:
        print(f"Wrote {path.relative_to(REPOSITORY_ROOT)}")
    print(f"\n{len(written)} synthetic instance(s) materialized in data/synthetic/.")


if __name__ == "__main__":
    main()
