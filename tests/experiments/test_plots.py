"""Tests for the plotting helpers."""

from src.experiments import layout
from src.experiments.plots import _load_convergence, _ordered_configs

_CONVERGENCE_FIELDS = ["instance", "config", "seed", "generation", "best_makespan"]


def test_ordered_configs_uses_canonical_order_and_drops_unknown() -> None:
    ordered = _ordered_configs(["pmx_cpm", "ox", "unknown", "pmx"])

    assert ordered == ["ox", "pmx", "pmx_cpm"]


def test_load_convergence_averages_across_seeds(tmp_path) -> None:
    campaign = tmp_path / "camp"
    rows = [
        {
            "instance": "inst",
            "config": "ox",
            "seed": 0,
            "generation": 0,
            "best_makespan": 10,
        },
        {
            "instance": "inst",
            "config": "ox",
            "seed": 0,
            "generation": 1,
            "best_makespan": 8,
        },
        {
            "instance": "inst",
            "config": "ox",
            "seed": 1,
            "generation": 0,
            "best_makespan": 12,
        },
        {
            "instance": "inst",
            "config": "ox",
            "seed": 1,
            "generation": 1,
            "best_makespan": 6,
        },
    ]
    layout.write_csv(
        layout.cell_dir(campaign, "inst", "ox") / "convergence.csv",
        _CONVERGENCE_FIELDS,
        rows,
    )

    curves = _load_convergence(campaign)

    assert curves["inst"]["ox"] == [11.0, 7.0]
