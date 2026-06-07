"""Tests for the shared campaign layout helpers."""

import csv

from src.experiments import layout


def test_cell_label_encodes_cpm() -> None:
    assert layout.cell_label("ox", False) == "ox"
    assert layout.cell_label("ox", True) == "ox_cpm"
    assert layout.cell_label("pmx", True) == "pmx_cpm"


def test_cell_and_baseline_paths_compose(tmp_path) -> None:
    campaign = tmp_path / "camp"

    cell = layout.cell_dir(campaign, "syn_t15_p010", "ox_cpm")
    assert cell == campaign / "runs" / "syn_t15_p010__ox_cpm"
    assert layout.baseline_path(campaign, "syn_t15_p010") == (
        campaign / "baselines" / "syn_t15_p010.json"
    )


def test_resolve_campaign_dir_prefers_explicit(tmp_path) -> None:
    explicit = tmp_path / "explicit"

    assert layout.resolve_campaign_dir(explicit) == explicit
    assert explicit.is_dir()


def test_resolve_campaign_dir_uses_name(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(layout, "RESULTS_ROOT", tmp_path)

    resolved = layout.resolve_campaign_dir(name="smoke")

    assert resolved == tmp_path / "smoke"
    assert resolved.is_dir()


def test_json_round_trip(tmp_path) -> None:
    path = tmp_path / "x.json"
    payload = {"a": 1, "b": [1, 2], "c": None}

    layout.write_json(path, payload)

    assert layout.read_json(path) == payload


def test_csv_round_trip(tmp_path) -> None:
    path = tmp_path / "x.csv"

    layout.write_csv(path, ["a", "b"], [{"a": 1, "b": 2}, {"a": 3, "b": 4}])

    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [{"a": "1", "b": "2"}, {"a": "3", "b": "4"}]
