"""Plots for the experiment campaign.

Reads ``runs.csv`` and ``convergence.csv`` written by ``run_experiments`` and
saves, per instance, a boxplot of best ``C_max`` across configurations and a mean
convergence curve. Figures are written to ``<input-dir>/plots``.

Run from the repository root after ``run_experiments``::

    python -m src.experiments.plots
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIR = REPOSITORY_ROOT / "results" / "campaign"
CONFIG_ORDER = ("ox", "pmx", "ox_cpm", "pmx_cpm")


def main(argv: list[str] | None = None) -> None:
    """Generates boxplot and convergence figures for every instance."""
    arguments = _parse_arguments(argv)
    plt.switch_backend("Agg")
    plots_dir = arguments.input_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    best_makespans = _load_runs(arguments.input_dir / "runs.csv")
    convergence = _load_convergence(arguments.input_dir / "convergence.csv")

    for instance, config_values in best_makespans.items():
        _plot_boxplot(instance, config_values, plots_dir / f"{instance}_boxplot.png")
    for instance, config_curves in convergence.items():
        _plot_convergence(
            instance, config_curves, plots_dir / f"{instance}_convergence.png"
        )

    figure_count = len(best_makespans) + len(convergence)
    print(f"Wrote {figure_count} figures to {plots_dir}")


def _load_runs(path: Path) -> dict[str, dict[str, list[int]]]:
    """Loads best makespans grouped by instance then configuration."""
    grouped: dict[str, dict[str, list[int]]] = {}
    with path.open(encoding="utf-8") as runs_file:
        for row in csv.DictReader(runs_file):
            instance = grouped.setdefault(row["instance"], {})
            instance.setdefault(row["config"], []).append(int(row["best_makespan"]))
    return grouped


def _load_convergence(path: Path) -> dict[str, dict[str, list[float]]]:
    """Loads mean convergence curves grouped by instance then configuration."""
    indexed: dict[str, dict[str, dict[int, float]]] = {}
    with path.open(encoding="utf-8") as convergence_file:
        for row in csv.DictReader(convergence_file):
            instance = indexed.setdefault(row["instance"], {})
            config = instance.setdefault(row["config"], {})
            config[int(row["generation"])] = float(row["mean_best_makespan"])

    return {
        instance: {
            config: [values[generation] for generation in sorted(values)]
            for config, values in configs.items()
        }
        for instance, configs in indexed.items()
    }


def _ordered_configs(config_names: list[str]) -> list[str]:
    """Returns present configurations in the canonical display order."""
    return [config for config in CONFIG_ORDER if config in config_names]


def _plot_boxplot(
    instance: str,
    config_values: dict[str, list[int]],
    path: Path,
) -> None:
    """Saves a boxplot of best makespans per configuration."""
    labels = _ordered_configs(list(config_values))
    data = [config_values[config] for config in labels]

    figure, axes = plt.subplots()
    axes.boxplot(data, tick_labels=labels)
    axes.set_title(f"Best C_max by configuration — {instance}")
    axes.set_ylabel("best C_max")
    figure.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(figure)


def _plot_convergence(
    instance: str,
    config_curves: dict[str, list[float]],
    path: Path,
) -> None:
    """Saves mean convergence curves per configuration."""
    figure, axes = plt.subplots()
    for config in _ordered_configs(list(config_curves)):
        curve = config_curves[config]
        axes.plot(range(len(curve)), curve, label=config)

    axes.set_title(f"Mean best C_max per generation — {instance}")
    axes.set_xlabel("generation")
    axes.set_ylabel("mean best C_max")
    axes.legend()
    figure.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(figure)


def _parse_arguments(argv: list[str] | None) -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Plot the experiment campaign.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_DIR,
        help="Directory containing runs.csv and convergence.csv.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    main()
