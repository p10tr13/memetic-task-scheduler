"""Tests for the benchmark instance set."""

from src.experiments.instances import build_instances


def test_build_instances_is_deterministic() -> None:
    first = build_instances()
    second = build_instances()

    assert [entry.name for entry in first] == [entry.name for entry in second]
    for first_entry, second_entry in zip(first, second):
        assert first_entry.metadata.nodes == second_entry.metadata.nodes
        assert first_entry.metadata.edges == second_entry.metadata.edges


def test_build_instances_names_are_unique() -> None:
    names = [entry.name for entry in build_instances()]

    assert len(names) == len(set(names))


def test_small_instances_support_milp_and_large_ones_do_not() -> None:
    instances = {entry.name: entry for entry in build_instances()}

    assert instances["syn_t15_p010"].supports_milp
    assert instances["syn_t22_p018"].supports_milp
    assert not instances["rand0000_t300_p016"].supports_milp


def test_density_classes_straddle_the_sparse_threshold() -> None:
    instances = {entry.name: entry for entry in build_instances()}

    assert instances["syn_t40_p006"].metadata.density < 0.1
    assert instances["syn_t40_p035"].metadata.density >= 0.1


def test_quick_set_is_small() -> None:
    quick = build_instances(quick=True)

    assert quick
    assert all(entry.seed_count <= 3 for entry in quick)
