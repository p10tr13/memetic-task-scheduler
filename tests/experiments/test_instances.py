"""Tests for the benchmark graph registry."""

import pytest

from src.experiments.instances import get_instance, list_instances


def test_get_instance_is_deterministic() -> None:
    first = get_instance("syn_t15_p010")
    second = get_instance("syn_t15_p010")

    assert first.metadata.nodes == second.metadata.nodes
    assert first.metadata.edges == second.metadata.edges


def test_full_instance_names_are_unique() -> None:
    names = list_instances()

    assert len(names) == len(set(names))


def test_small_instances_support_milp_and_large_ones_do_not() -> None:
    assert get_instance("syn_t15_p010").supports_milp
    assert get_instance("syn_t22_p018").supports_milp
    assert not get_instance("rand0000_t300_p016").supports_milp


def test_density_classes_straddle_the_sparse_threshold() -> None:
    assert get_instance("syn_t40_p006").metadata.density < 0.1
    assert get_instance("syn_t40_p035").metadata.density >= 0.1


def test_quick_set_is_small_and_separate() -> None:
    quick = list_instances(quick=True)

    assert quick
    assert set(quick).isdisjoint(list_instances())
    assert all(get_instance(name).metadata.nodes <= 25 for name in quick)


def test_get_instance_rejects_unknown_name() -> None:
    with pytest.raises(KeyError):
        get_instance("does_not_exist")
