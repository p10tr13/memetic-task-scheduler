"""Tests for the campaign runner's job building and sub-cell merge."""

from src.experiments import layout, run_campaign


def test_build_jobs_expands_per_seed_units() -> None:
    args = run_campaign._parse_arguments(
        ["--per-seed", "--seeds", "3", "--instances", "a", "b"]
    )
    seeds = run_campaign._campaign_seeds(args)

    jobs = run_campaign._build_jobs(args, layout.RESULTS_ROOT, ["a", "b"], seeds)

    # 2 instances x 4 methods x 3 seeds.
    assert len(jobs) == 2 * len(run_campaign.METHODS) * 3
    labels = {label for _command, label in jobs}
    assert "a/ox#0" in labels
    assert "b/pmx_cpm#2" in labels


def test_build_jobs_without_per_seed_is_one_unit_per_cell() -> None:
    args = run_campaign._parse_arguments(["--instances", "a"])
    seeds = run_campaign._campaign_seeds(args)

    jobs = run_campaign._build_jobs(args, layout.RESULTS_ROOT, ["a"], seeds)

    assert len(jobs) == len(run_campaign.METHODS)
    assert all("#" not in label for _command, label in jobs)


def test_cell_flags_forward_budget_and_subcell() -> None:
    args = run_campaign._parse_arguments(["--per-seed", "--equal-budget"])

    flags = run_campaign._cell_flags(
        args, layout.RESULTS_ROOT, "a", {"crossover": "ox", "cpm": True}, 4
    )

    assert "--cpm" in flags
    assert "--equal-budget" in flags
    assert flags[flags.index("--seed-start") + 1] == "4"
    assert "--sub-cell" in flags


def test_merge_subcells_concatenates_seed_rows(tmp_path) -> None:
    campaign = tmp_path / "camp"
    method = {"crossover": "ox", "cpm": False}
    label = layout.cell_label(method["crossover"], method["cpm"])
    for seed in (0, 1):
        sub = layout.subcell_dir(campaign, "inst", label, seed)
        run_campaign.run_single.write_cell(
            sub,
            {"label": label, "seeds": 1, "seed_start": seed},
            [
                {
                    "instance": "inst",
                    "config": label,
                    "seed": seed,
                    "best_makespan": 10 + seed,
                }
            ],
            [
                {
                    "instance": "inst",
                    "config": label,
                    "seed": seed,
                    "generation": 0,
                    "best_makespan": 12,
                }
            ],
            {"per_seed_seconds": [1.0], "per_seed_cpu_seconds": [0.5]},
            [
                {
                    "instance": "inst",
                    "config": label,
                    "seed": seed,
                    "evaluations": 50,
                    "best_makespan": 10 + seed,
                }
            ],
        )

    run_campaign._merge_subcells(campaign, ["inst"], [0, 1])

    cell = layout.cell_dir(campaign, "inst", label)
    merged = run_campaign._read_csv(cell / "runs.csv")
    assert [row["seed"] for row in merged] == ["0", "1"]
    config = layout.read_json(cell / "config.json")
    assert config["seeds"] == 2
    timing = layout.read_json(cell / "timing.json")
    assert timing["total_seconds"] == 2.0
    assert timing["merged_from_subcells"] == 2
