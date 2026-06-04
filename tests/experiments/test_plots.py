"""Tests for the plotting helpers."""

from src.experiments.plots import _ordered_configs


def test_ordered_configs_uses_canonical_order_and_drops_unknown() -> None:
    ordered = _ordered_configs(["pmx_cpm", "ox", "unknown", "pmx"])

    assert ordered == ["ox", "pmx", "pmx_cpm"]
