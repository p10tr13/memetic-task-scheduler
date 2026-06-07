"""Domain models for task scheduling instances."""

import weakref
from dataclasses import dataclass

import networkx as nx
import numpy as np


@dataclass(frozen=True)
class ProblemInstance:
    """Stores a task scheduling problem instance.

    Args:
        number_of_groups: Number of available processing groups, if known.
        priority_relationship: Directed graph of task precedence constraints.
        task_times: Processing times indexed by task number.
    """

    number_of_groups: int | None
    priority_relationship: nx.DiGraph
    task_times: np.ndarray


# Plain dict adjacency cached per graph object. networkx ``.predecessors()``
# rebuilds an iterator over the adjacency view on every call; the precedence
# graph never changes during a run, so the hot decoder/CPM loops index this
# precomputed ``dict[int, list[int]]`` instead. Keyed weakly so caches vanish
# with their graphs; the instance is frozen, hence not a new field.
_PREDECESSORS_CACHE: "weakref.WeakKeyDictionary[nx.DiGraph, dict[int, list[int]]]" = (
    weakref.WeakKeyDictionary()
)


def predecessors_map(graph: nx.DiGraph) -> dict[int, list[int]]:
    """Returns a cached task -> predecessor-list mapping for the graph.

    Args:
        graph: Precedence graph whose predecessor lists are needed.

    Returns:
        Mapping from each task to the list of its direct predecessors. The same
        cached object is returned for repeated calls on the same graph.
    """
    cached = _PREDECESSORS_CACHE.get(graph)
    if cached is None:
        cached = {node: list(graph.predecessors(node)) for node in graph.nodes}
        _PREDECESSORS_CACHE[graph] = cached
    return cached
