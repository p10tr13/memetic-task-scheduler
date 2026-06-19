"""Tests for the campaign analysis helpers."""

from src.experiments import layout
from src.experiments.analyze import (
    _h1_rows,
    _h3_rows,
    _holm_adjusted,
    _load_baselines,
    _load_runs,
    _wilcoxon_p,
)


def test_h3_rows_flag_tuned_improvement() -> None:
    # 15 seeds, as in the real campaign (n must be large enough for Wilcoxon to
    # be able to reach significance at all).
    tuned = {
        ("inst", "pmx_cpm"): [90] * 15,
        ("inst", "ox"): [50] * 15,
    }
    default = {
        ("inst", "pmx_cpm"): [100] * 15,
        ("inst", "ox"): [50] * 15,
    }

    rows = {row["config"]: row for row in _h3_rows(tuned, default)}

    assert rows["pmx_cpm"]["better"] == "tuned"
    assert rows["pmx_cpm"]["delta_median"] == 10
    assert rows["pmx_cpm"]["significant"] is True
    # An all-equal cell is a tie and not significant.
    assert rows["ox"]["better"] == "tie"
    assert rows["ox"]["significant"] is False


def test_h3_rows_skip_cells_missing_from_default() -> None:
    tuned = {("inst", "pmx"): [1, 2, 3]}
    default = {("other", "pmx"): [1, 2, 3]}

    assert _h3_rows(tuned, default) == []


_RUN_FIELDS = [
    "instance",
    "config",
    "crossover",
    "use_cpm",
    "seed",
    "best_makespan",
    "processors",
    "nodes",
    "edges",
    "density",
]


def test_holm_adjusted_is_monotone_and_scaled() -> None:
    adjusted = _holm_adjusted([0.01, 0.04, 0.03])

    assert adjusted == [0.03, 0.06, 0.06]


def test_wilcoxon_p_is_one_for_identical_samples() -> None:
    assert _wilcoxon_p([5, 5, 5, 5], [5, 5, 5, 5]) == 1.0


def test_wilcoxon_p_is_small_for_consistent_difference() -> None:
    worse = [10, 10, 10, 10, 10, 10, 10, 10]
    better = [5, 5, 5, 5, 5, 5, 5, 5]

    assert _wilcoxon_p(worse, better) < 0.05


def _write_cell(campaign, instance, label, makespans) -> None:
    """Writes a minimal runs.csv cell for the fixture campaign."""
    rows = [
        {
            "instance": instance,
            "config": label,
            "crossover": label.split("_")[0],
            "use_cpm": "_cpm" in label,
            "seed": seed,
            "best_makespan": value,
            "processors": 2,
            "nodes": 5,
            "edges": 4,
            "density": 0.05,
        }
        for seed, value in enumerate(makespans)
    ]
    layout.write_csv(
        layout.cell_dir(campaign, instance, label) / "runs.csv", _RUN_FIELDS, rows
    )


def test_loaders_and_h1_over_fixture_campaign(tmp_path) -> None:
    campaign = tmp_path / "camp"
    _write_cell(campaign, "inst", "ox", [10, 10, 11, 10])
    _write_cell(campaign, "inst", "ox_cpm", [8, 8, 9, 8])
    layout.write_json(
        layout.baseline_path(campaign, "inst"),
        {
            "instance": "inst",
            "density": 0.05,
            "lower_bound": 7,
            "milp_makespan": 7,
            "milp_is_optimal": True,
        },
    )

    runs = _load_runs(campaign)
    assert runs[("inst", "ox")] == [10, 10, 11, 10]
    assert runs[("inst", "ox_cpm")] == [8, 8, 9, 8]

    baselines = _load_baselines(campaign)
    assert baselines["inst"]["milp_is_optimal"] is True

    h1 = _h1_rows(runs)
    ox_row = next(row for row in h1 if row["crossover"] == "ox")
    assert ox_row["delta_median"] > 0
