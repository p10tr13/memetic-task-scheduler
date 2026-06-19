"""Tests for the DEAP evolutionary algorithm."""

import networkx as nx
import numpy as np
import pytest

from src.core.decoder import calculate_makespan
from src.core.models import ProblemInstance
from src.solvers.evolutionary.algorithm import (
    EvolutionConfig,
    EvolutionStrategies,
    evaluate_permutation_fitness,
    null_crossover,
    null_local_search,
    run_evolution,
    swap_mutation,
)


def test_run_evolution_returns_valid_result() -> None:
    problem_instance = _create_independent_instance()

    result = run_evolution(
        problem_instance=problem_instance,
        config=EvolutionConfig(
            population_size=6,
            generations=3,
            crossover_probability=0.0,
            mutation_probability=0.0,
            random_seed=1,
        ),
    )

    assert sorted(result.best_permutation) == [0, 1, 2]
    assert result.best_makespan == calculate_makespan(
        result.best_permutation,
        problem_instance,
    )
    assert len(result.generation_best_makespans) == 4


def test_run_evolution_tracks_evaluations() -> None:
    problem_instance = _create_independent_instance()

    result = run_evolution(
        problem_instance=problem_instance,
        config=EvolutionConfig(
            population_size=6,
            generations=3,
            crossover_probability=1.0,
            mutation_probability=1.0,
            random_seed=1,
        ),
    )

    # The initial population is fully evaluated, so the first eval-curve sample
    # is exactly the population size, and one sample is recorded per generation
    # boundary (initial + each generation).
    assert result.evaluation_best_makespans[0][0] == 6
    assert len(result.evaluation_best_makespans) == len(
        result.generation_best_makespans
    )
    evaluations = [evals for evals, _ in result.evaluation_best_makespans]
    makespans = [makespan for _, makespan in result.evaluation_best_makespans]
    assert evaluations == sorted(evaluations)
    assert makespans == sorted(makespans, reverse=True)
    assert result.total_evaluations == evaluations[-1] >= 6


def test_run_evolution_stops_at_evaluation_budget() -> None:
    problem_instance = _create_independent_instance()
    shared = dict(
        population_size=6,
        crossover_probability=1.0,
        mutation_probability=1.0,
        random_seed=4,
    )

    uncapped = run_evolution(
        problem_instance=problem_instance,
        config=EvolutionConfig(generations=50, **shared),
    )
    capped = run_evolution(
        problem_instance=problem_instance,
        config=EvolutionConfig(generations=50, max_evaluations=20, **shared),
    )

    # The budget stops the run far short of the generation cap, after the first
    # generation boundary that reaches the budget.
    assert capped.total_evaluations >= 20
    assert capped.total_evaluations < uncapped.total_evaluations
    assert len(capped.generation_best_makespans) < 51


def test_run_evolution_uses_configured_strategies() -> None:
    problem_instance = _create_independent_instance()
    strategy_calls = {
        "selection": 0,
        "mutation": 0,
        "crossover": 0,
        "local_search": 0,
    }

    def selection_strategy(population, selection_size):
        strategy_calls["selection"] += 1
        return list(population[:selection_size])

    def mutation_strategy(individual):
        strategy_calls["mutation"] += 1
        individual[0], individual[1] = individual[1], individual[0]
        return (individual,)

    def crossover_strategy(first_individual, second_individual):
        strategy_calls["crossover"] += 1
        first_individual[0], second_individual[0] = (
            second_individual[0],
            first_individual[0],
        )
        return first_individual, second_individual

    def local_search_strategy(problem_instance, task_permutation):
        strategy_calls["local_search"] += 1
        return list(task_permutation)

    run_evolution(
        problem_instance=problem_instance,
        config=EvolutionConfig(
            population_size=4,
            generations=1,
            crossover_probability=1.0,
            mutation_probability=1.0,
            random_seed=2,
        ),
        strategies=EvolutionStrategies(
            selection_strategy=selection_strategy,
            mutation_strategy=mutation_strategy,
            crossover_strategy=crossover_strategy,
            local_search_strategy=local_search_strategy,
        ),
    )

    assert strategy_calls == {
        "selection": 1,
        "mutation": 4,
        "crossover": 2,
        "local_search": 4,
    }


def test_run_evolution_handles_local_search_value_error() -> None:
    problem_instance = _create_independent_instance()

    def local_search_strategy(problem_instance, task_permutation):
        raise ValueError("Invalid local search candidate.")

    result = run_evolution(
        problem_instance=problem_instance,
        config=EvolutionConfig(
            population_size=4,
            generations=1,
            crossover_probability=0.0,
            mutation_probability=0.0,
            random_seed=3,
        ),
        strategies=EvolutionStrategies(local_search_strategy=local_search_strategy),
    )

    assert sorted(result.best_permutation) == [0, 1, 2]


def test_evaluate_permutation_fitness_returns_decoder_makespan() -> None:
    problem_instance = _create_independent_instance()

    fitness = evaluate_permutation_fitness(
        [0, 1, 2],
        problem_instance,
        invalid_makespan_penalty=999,
    )

    assert fitness == (4,)


def test_evaluate_permutation_fitness_penalizes_invalid_permutation() -> None:
    problem_instance = _create_dependent_instance()

    fitness = evaluate_permutation_fitness(
        [1, 0, 2],
        problem_instance,
        invalid_makespan_penalty=999,
    )

    assert fitness == (999,)


def test_swap_mutation_preserves_tasks() -> None:
    individual = [0, 1, 2, 3]

    mutated_individual = swap_mutation(individual)[0]

    assert sorted(mutated_individual) == [0, 1, 2, 3]


def test_null_strategies_leave_individuals_unchanged() -> None:
    first_individual = [0, 1]
    second_individual = [1, 0]

    crossed_individuals = null_crossover(first_individual, second_individual)
    searched_individual = null_local_search(
        _create_independent_instance(),
        first_individual,
    )

    assert crossed_individuals == ([0, 1], [1, 0])
    assert searched_individual == [0, 1]
    assert searched_individual is not first_individual


def test_run_evolution_rejects_invalid_parameters() -> None:
    with pytest.raises(ValueError, match="Population size"):
        run_evolution(
            problem_instance=_create_independent_instance(),
            config=EvolutionConfig(
                population_size=0,
                generations=1,
                crossover_probability=0.0,
                mutation_probability=0.0,
            ),
        )


def _create_independent_instance() -> ProblemInstance:
    """Creates an instance where all task permutations are valid.

    Returns:
        ProblemInstance with independent tasks.
    """
    graph = nx.DiGraph()
    graph.add_nodes_from([0, 1, 2])

    return ProblemInstance(
        number_of_groups=2,
        priority_relationship=graph,
        task_times=np.array([1, 2, 3]),
    )


def _create_dependent_instance() -> ProblemInstance:
    """Creates an instance with precedence constraints.

    Returns:
        ProblemInstance with a simple task chain.
    """
    graph = nx.DiGraph()
    graph.add_edges_from([(0, 1), (1, 2)])

    return ProblemInstance(
        number_of_groups=1,
        priority_relationship=graph,
        task_times=np.array([1, 2, 3]),
    )
