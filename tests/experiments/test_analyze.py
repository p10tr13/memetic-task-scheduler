"""Tests for the campaign analysis helpers."""

from src.experiments.analyze import _holm_adjusted, _wilcoxon_p


def test_holm_adjusted_is_monotone_and_scaled() -> None:
    adjusted = _holm_adjusted([0.01, 0.04, 0.03])

    assert adjusted == [0.03, 0.06, 0.06]


def test_wilcoxon_p_is_one_for_identical_samples() -> None:
    assert _wilcoxon_p([5, 5, 5, 5], [5, 5, 5, 5]) == 1.0


def test_wilcoxon_p_is_small_for_consistent_difference() -> None:
    worse = [10, 10, 10, 10, 10, 10, 10, 10]
    better = [5, 5, 5, 5, 5, 5, 5, 5]

    assert _wilcoxon_p(worse, better) < 0.05
