"""DEAP setup helpers for evolutionary scheduling."""

import random
import networkx as nx

from deap import base, creator, tools

from src.core.models import ProblemInstance
from src.solvers.evolutionary.config import EvolutionConfig, Individual
from src.solvers.evolutionary.strategies import evaluate_permutation_fitness


def ensure_deap_classes() -> None:
    """Creates DEAP classes required by the evolutionary algorithm."""
    if not hasattr(creator, "TaskSchedulingFitnessMin"):
        creator.create("TaskSchedulingFitnessMin", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, "TaskSchedulingIndividual"):
        fitness_class = getattr(creator, "TaskSchedulingFitnessMin")
        creator.create(
            "TaskSchedulingIndividual",
            list,
            fitness=fitness_class,
        )


def create_toolbox(
    problem_instance: ProblemInstance,
    config: EvolutionConfig,
) -> base.Toolbox:
    """Creates a DEAP toolbox for one evolutionary run.

    Args:
        problem_instance: Scheduling problem instance.
        config: Evolutionary algorithm configuration.

    Returns:
        Configured DEAP toolbox.
    """
    individual_class = getattr(creator, "TaskSchedulingIndividual")
    penalty = config.invalid_makespan_penalty or _calculate_default_penalty(
        problem_instance
    )
    toolbox = base.Toolbox()
    toolbox.register(
        "individual",
        _create_individual,
        individual_class,
        problem_instance,
        _create_task_numbers(problem_instance),
    )
    toolbox.register(
        "population",
        tools.initRepeat,
        list,
        getattr(toolbox, "individual"),
    )
    toolbox.register(
        "evaluate",
        evaluate_permutation_fitness,
        problem_instance=problem_instance,
        invalid_makespan_penalty=penalty,
    )
    toolbox.register("clone", _clone_individual, individual_class)

    return toolbox


def evaluate_population(population: list[Individual], toolbox: base.Toolbox) -> None:
    """Evaluates all individuals with invalid fitness.

    Args:
        population: Population to evaluate.
        toolbox: DEAP toolbox with evaluation function.
    """
    invalid_individuals = [
        individual for individual in population if not individual.fitness.valid
    ]

    evaluate = getattr(toolbox, "evaluate")
    for individual, fitness_values in zip(
        invalid_individuals,
        map(evaluate, invalid_individuals),
    ):
        individual.fitness.values = fitness_values


def _create_task_numbers(problem_instance: ProblemInstance) -> list[int]:
    """Creates the sorted task numbers represented by each individual.

    Args:
        problem_instance: Scheduling problem instance.

    Returns:
        Sorted task numbers from the precedence graph.
    """
    return sorted(problem_instance.priority_relationship.nodes)


def _create_individual(
    individual_class: type,
    problem_instance: ProblemInstance,
    task_numbers: list[int],
) -> Individual:
    """Creates a random topological task permutation.

    Args:
        individual_class: DEAP individual class.
        problem_instance: Scheduling problem instance.
        task_numbers: Tasks represented by the individual.

    Returns:
        DEAP individual containing every task once.
    """
    permutation: list[int] = []

    for generation in nx.topological_generations(problem_instance.priority_relationship):
        generation_tasks = sorted(task for task in generation if task in task_numbers)
        random.shuffle(generation_tasks)
        permutation.extend(generation_tasks)

    return individual_class(permutation)


def _clone_individual(individual_class: type, individual: Individual) -> Individual:
    """Clones a DEAP individual and its fitness.

    Args:
        individual_class: DEAP individual class.
        individual: Individual to clone.

    Returns:
        Copied individual.
    """
    cloned_individual = individual_class(individual)
    cloned_individual.fitness.values = individual.fitness.values

    return cloned_individual


def _calculate_default_penalty(problem_instance: ProblemInstance) -> int:
    """Returns a large penalty for infeasible schedules.

    Args:
        problem_instance: Scheduling problem instance.

    Returns:
        Penalty larger than a serial execution of all task times.
    """
    return int(sum(problem_instance.task_times)) * len(problem_instance.task_times) + 1
