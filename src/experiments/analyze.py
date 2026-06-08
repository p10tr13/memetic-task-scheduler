"""Statistical analysis of an experiment campaign.

Reads the per-cell ``runs.csv`` and per-instance ``baselines/*.json`` written
under a campaign directory and produces:

- ``summary.csv`` — best/mean/median/std and the optimality gap per
  instance x configuration (gap vs the MILP optimum where proven, otherwise vs
  the lower bound);
- ``h1_wilcoxon.csv`` — paired Wilcoxon tests of CPM-on vs CPM-off (per crossover
  per instance), Holm-corrected, testing hypothesis H1;
- ``h2_wilcoxon.csv`` — paired Wilcoxon tests of OX vs PMX (pure EA, per
  instance), Holm-corrected, testing hypothesis H2 with sparse/dense labels.

Run from the repository root, pointing at a campaign directory::

    python -m src.experiments.analyze --campaign results/0607_1105
"""

import argparse
import csv
import statistics
from pathlib import Path

from scipy.stats import wilcoxon

from src.experiments import layout

SIGNIFICANCE_LEVEL = 0.05
SPARSE_DENSITY_THRESHOLD = 0.1

SUMMARY_FIELDS = [
    "instance",
    "config",
    "best",
    "mean",
    "median",
    "std",
    "reference",
    "reference_kind",
    "gap_percent",
]
H1_FIELDS = [
    "instance",
    "crossover",
    "median_cpm_off",
    "median_cpm_on",
    "delta_median",
    "wilcoxon_p",
    "holm_p",
    "significant",
]
H2_FIELDS = [
    "instance",
    "density",
    "density_class",
    "median_ox",
    "median_pmx",
    "delta_median",
    "better",
    "wilcoxon_p",
    "holm_p",
    "significant",
]


def main(argv: list[str] | None = None) -> None:
    """Analyses a campaign directory and writes summary and hypothesis tables."""
    arguments = _parse_arguments(argv)
    campaign = arguments.campaign
    runs = _load_runs(campaign)
    baselines = _load_baselines(campaign)

    summary_rows = _summary_rows(runs, baselines)
    h1_rows = _h1_rows(runs)
    h2_rows = _h2_rows(runs, baselines)

    layout.write_csv(campaign / "summary.csv", SUMMARY_FIELDS, summary_rows)
    layout.write_csv(campaign / "h1_wilcoxon.csv", H1_FIELDS, h1_rows)
    layout.write_csv(campaign / "h2_wilcoxon.csv", H2_FIELDS, h2_rows)

    _print_overview(h1_rows, h2_rows)
    print(f"Wrote summary/h1_wilcoxon/h2_wilcoxon to {campaign}")


def _load_runs(campaign: Path) -> dict[tuple[str, str], list[int]]:
    """Loads best makespans keyed by (instance, config), ordered by seed.

    Walks every ``runs/<instance>__<label>/runs.csv`` cell in the campaign.
    """
    grouped: dict[tuple[str, str], dict[int, int]] = {}
    for runs_csv in sorted(layout.runs_dir(campaign).glob("*/runs.csv")):
        with runs_csv.open(encoding="utf-8") as runs_file:
            for row in csv.DictReader(runs_file):
                key = (row["instance"], row["config"])
                grouped.setdefault(key, {})[int(row["seed"])] = int(
                    row["best_makespan"]
                )

    return {
        key: [makespans[seed] for seed in sorted(makespans)]
        for key, makespans in grouped.items()
    }


def _load_baselines(campaign: Path) -> dict[str, dict]:
    """Loads the baseline payload per instance from ``baselines/*.json``."""
    baselines: dict[str, dict] = {}
    for path in sorted(layout.baselines_dir(campaign).glob("*.json")):
        row = layout.read_json(path)
        baselines[row["instance"]] = row
    return baselines


def _instances_in_order(runs: dict[tuple[str, str], list[int]]) -> list[str]:
    """Returns instance names in first-seen order."""
    ordered: list[str] = []
    for instance, _config in runs:
        if instance not in ordered:
            ordered.append(instance)
    return ordered


def _summary_rows(
    runs: dict[tuple[str, str], list[int]],
    baselines: dict[str, dict],
) -> list[dict]:
    """Builds best/mean/median/std and gap rows per instance x config."""
    rows: list[dict] = []
    for (instance, config), makespans in runs.items():
        reference, reference_kind = _reference(baselines[instance])
        median = statistics.median(makespans)
        rows.append(
            {
                "instance": instance,
                "config": config,
                "best": min(makespans),
                "mean": round(statistics.mean(makespans), 2),
                "median": median,
                "std": round(statistics.pstdev(makespans), 2),
                "reference": reference,
                "reference_kind": reference_kind,
                "gap_percent": _gap_percent(median, reference),
            }
        )
    return rows


def _reference(baseline_row: dict) -> tuple[int, str]:
    """Returns the reference makespan and its kind (MILP optimum or bound)."""
    if (
        baseline_row.get("milp_is_optimal")
        and baseline_row.get("milp_makespan") is not None
    ):
        return int(baseline_row["milp_makespan"]), "milp"
    return int(baseline_row["lower_bound"]), "lower_bound"


def _gap_percent(value: float, reference: int) -> float:
    """Returns the percentage gap of a value above a reference."""
    if reference <= 0:
        return 0.0
    return round(100.0 * (value - reference) / reference, 2)


def _h1_rows(runs: dict[tuple[str, str], list[int]]) -> list[dict]:
    """Builds Holm-corrected CPM-on vs CPM-off rows (H1)."""
    rows: list[dict] = []
    for instance in _instances_in_order(runs):
        for crossover in ("ox", "pmx"):
            off = runs.get((instance, crossover))
            on = runs.get((instance, f"{crossover}_cpm"))
            if not off or not on:
                continue
            rows.append(
                {
                    "instance": instance,
                    "crossover": crossover,
                    "median_cpm_off": statistics.median(off),
                    "median_cpm_on": statistics.median(on),
                    "delta_median": statistics.median(off) - statistics.median(on),
                    "wilcoxon_p": round(_wilcoxon_p(off, on), 4),
                }
            )

    _apply_holm(rows)
    return rows


def _h2_rows(
    runs: dict[tuple[str, str], list[int]],
    baselines: dict[str, dict],
) -> list[dict]:
    """Builds Holm-corrected OX vs PMX rows (H2), labelled by density."""
    rows: list[dict] = []
    for instance in _instances_in_order(runs):
        ox_values = runs.get((instance, "ox"))
        pmx_values = runs.get((instance, "pmx"))
        if not ox_values or not pmx_values:
            continue
        density = float(baselines[instance]["density"])
        median_ox = statistics.median(ox_values)
        median_pmx = statistics.median(pmx_values)
        rows.append(
            {
                "instance": instance,
                "density": density,
                "density_class": _density_class(density),
                "median_ox": median_ox,
                "median_pmx": median_pmx,
                "delta_median": median_pmx - median_ox,
                "better": _better_operator(median_ox, median_pmx),
                "wilcoxon_p": round(_wilcoxon_p(ox_values, pmx_values), 4),
            }
        )

    _apply_holm(rows)
    return rows


def _density_class(density: float) -> str:
    """Classifies an instance as sparse or dense by the H2 threshold."""
    return "sparse" if density < SPARSE_DENSITY_THRESHOLD else "dense"


def _better_operator(median_ox: float, median_pmx: float) -> str:
    """Returns which crossover reached the lower median makespan."""
    if median_ox < median_pmx:
        return "ox"
    if median_pmx < median_ox:
        return "pmx"
    return "tie"


def _wilcoxon_p(values_a: list[int], values_b: list[int]) -> float:
    """Returns the paired Wilcoxon p-value, or 1.0 if the samples are equal."""
    differences = [a - b for a, b in zip(values_a, values_b)]
    if all(difference == 0 for difference in differences):
        return 1.0
    return float(wilcoxon(values_a, values_b).pvalue)


def _apply_holm(rows: list[dict]) -> None:
    """Adds Holm-adjusted p-values and significance flags to test rows."""
    raw_pvalues = [row["wilcoxon_p"] for row in rows]
    holm_pvalues = _holm_adjusted(raw_pvalues)
    for row, holm_p in zip(rows, holm_pvalues):
        row["holm_p"] = round(holm_p, 4)
        row["significant"] = holm_p < SIGNIFICANCE_LEVEL


def _holm_adjusted(pvalues: list[float]) -> list[float]:
    """Holm step-down adjustment of a family of p-values."""
    count = len(pvalues)
    ascending = sorted(range(count), key=lambda index: pvalues[index])
    adjusted = [0.0] * count
    running_max = 0.0
    for rank, index in enumerate(ascending):
        candidate = min(1.0, (count - rank) * pvalues[index])
        running_max = max(running_max, candidate)
        adjusted[index] = running_max
    return adjusted


def _print_overview(h1_rows: list[dict], h2_rows: list[dict]) -> None:
    """Prints a short human-readable verdict for H1 and H2."""
    h1_significant = sum(1 for row in h1_rows if row["significant"])
    h1_helps = sum(
        1 for row in h1_rows if row["significant"] and row["delta_median"] > 0
    )
    print(f"H1 (CPM): {h1_helps}/{h1_significant} significant tests favour CPM-on")

    sparse_ox = [
        row
        for row in h2_rows
        if row["density_class"] == "sparse" and row["significant"]
    ]
    sparse_ox_wins = sum(1 for row in sparse_ox if row["better"] == "ox")
    print(
        f"H2 (OX>PMX on sparse): {sparse_ox_wins}/{len(sparse_ox)} "
        "significant sparse tests favour OX"
    )


def _parse_arguments(argv: list[str] | None) -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Analyse the experiment campaign.")
    layout.add_campaign_args(parser, required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    main()
