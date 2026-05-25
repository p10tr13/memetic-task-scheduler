"""Domain models for task scheduling instances."""

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
