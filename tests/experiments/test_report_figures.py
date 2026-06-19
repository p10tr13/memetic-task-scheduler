"""Tests for the equal-budget convergence aggregation in report_figures."""

from src.experiments import report_figures


def test_best_at_step_interpolates_curve() -> None:
    curve = [(10, 100), (20, 90), (30, 80)]

    assert report_figures._best_at(curve, 5) == 100
    assert report_figures._best_at(curve, 25) == 90
    assert report_figures._best_at(curve, 100) == 80


def test_aggregate_convergence_evals_resamples_and_gaps() -> None:
    data = {
        "convergence_evals": {
            ("i1", "ox"): [
                [(10, 120), (50, 100), (100, 90)],
                [(10, 120), (50, 110), (100, 90)],
            ]
        },
        "reference": {"i1": 90},
    }

    rows = report_figures._aggregate_convergence_evals(data, points=3)

    # Grid spans the shared evaluation range (max first-eval .. min last-eval).
    assert [row[0] for row in rows] == [10, 55, 100]
    ox_gaps = [row[report_figures.CONFIG_ORDER.index("ox") + 1] for row in rows]
    # Mean best falls 120 -> ~100 -> 90, i.e. gap to LB=90 falls to 0.
    assert ox_gaps[0] > ox_gaps[-1]
    assert ox_gaps[-1] == 0.0


def test_aggregate_convergence_evals_empty_is_empty() -> None:
    assert (
        report_figures._aggregate_convergence_evals(
            {"convergence_evals": {}, "reference": {}}
        )
        == []
    )
