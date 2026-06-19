"""Smoke test for the Optuna tuning module."""

import pytest

pytest.importorskip("optuna")

from src.experiments import layout, tune  # noqa: E402


def test_tune_writes_best_params(tmp_path) -> None:
    out = tmp_path / "best_params.json"
    tune.main(
        [
            "--crossover",
            "ox",
            "--cpm",
            "--instances",
            "syn_t15_p010",
            "--tune-seeds",
            "1",
            "--n-trials",
            "2",
            "--quick",
            "--out",
            str(out),
        ]
    )

    payload = layout.read_json(out)
    assert payload["crossover"] == "ox"
    assert payload["use_cpm"] is True
    assert set(payload["params"]) == {
        "crossover_probability",
        "mutation_probability",
        "population_size",
        "tournament_size",
        "cpm_max_attempts",
    }
    assert payload["completed_trials"] >= 1
    assert "syn_t15_p010" in payload["tune_instances"]


def test_tune_pure_ea_omits_cpm_attempts(tmp_path) -> None:
    out = tmp_path / "best_params.json"
    tune.main(
        [
            "--crossover",
            "pmx",
            "--no-cpm",
            "--instances",
            "syn_t15_p010",
            "--tune-seeds",
            "1",
            "--n-trials",
            "1",
            "--quick",
            "--out",
            str(out),
        ]
    )

    payload = layout.read_json(out)
    assert "cpm_max_attempts" not in payload["params"]
    assert payload["use_cpm"] is False
