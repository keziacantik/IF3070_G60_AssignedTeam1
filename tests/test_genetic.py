from random import Random

import pytest

from src.algorithms.genetic import GeneticAlgorithm
from src.core.constraints import Feasibility
from src.core.initializer import random_state
from src.core.io import load_problem
from src.core.neighborhood import Neighborhood
from src.core.objective import TotalShippingFee


@pytest.fixture(scope="module")
def problem():
    return load_problem("data/medium.json")


def make(**params):
    return GeneticAlgorithm(Neighborhood(), TotalShippingFee(), **params)


def run(problem, seed, **params):
    params = {"population_size": 10, "iterations": 15, **params}
    rng = Random(seed)
    initial = random_state(problem, rng)
    return make(**params).run(initial, rng)


@pytest.fixture(scope="module")
def result(problem):
    return run(problem, 1)


def test_result_is_feasible(result):
    assert Feasibility().is_feasible(result.final_state)


def test_one_record_per_generation(result):
    n = result.extra["parameters"]["iterations"]
    assert result.iterations == len(result.history) == len(result.extra["best_history"]) == n
    # index 0 of the population curves is the initial population
    assert len(result.extra["max_per_generation"]) == len(result.extra["mean_per_generation"]) == n + 1
    assert result.extra["max_per_generation"][1:] == result.history


def test_same_seed_same_result(problem):
    a, b = run(problem, 3), run(problem, 3)
    assert a.final_state == b.final_state
    assert a.history == b.history
    assert a.extra == b.extra


def test_returns_best_individual_seen(result):
    extra = result.extra
    best_history = extra["best_history"]
    assert all(b >= a for a, b in zip(best_history, best_history[1:]))
    assert result.final_value == extra["best_value"] == best_history[-1]
    assert extra["best_value"] == max(extra["max_per_generation"])
    assert result.final_value >= result.initial_value


def test_max_is_at_least_mean(result):
    extra = result.extra
    assert all(m >= a for m, a in zip(extra["max_per_generation"], extra["mean_per_generation"]))


def test_crossover_children_are_feasible(problem):
    ga, rng, feasibility = make(), Random(5), Feasibility()
    for _ in range(20):
        mother, father = random_state(problem, rng), random_state(problem, rng)
        for child in ga._crossover(mother, father, rng):
            assert feasibility.is_feasible(child)


def test_fill_keeps_loaded_vehicles_and_stays_feasible(problem):
    ga, rng, feasibility = make(), Random(6), Feasibility()
    for _ in range(10):
        state = random_state(problem, rng, tries_per_vehicle=1)
        filled = ga._fill(state, rng)
        assert feasibility.is_feasible(filled)
        assert all(filled.pos[i] == state.pos[i] for i in state.placed_indices())


def test_fill_disabled_is_identity(problem):
    state = random_state(problem, Random(7))
    assert make(fill_tries=0)._fill(state, Random(0)) == state


def test_roulette_handles_zero_fitness(problem):
    population = [random_state(problem, Random(s)) for s in range(3)]
    assert GeneticAlgorithm._select(population, [0, 0, 0], Random(0)) in population


def test_roulette_never_picks_zero_fitness(problem):
    population = [random_state(problem, Random(s)) for s in range(3)]
    rng = Random(0)
    picks = {GeneticAlgorithm._select(population, [0, 10, 0], rng) for _ in range(50)}
    assert picks == {population[1]}


@pytest.mark.parametrize("params", [
    {"population_size": 1}, {"iterations": 0},
    {"mutation_rate": -0.1}, {"mutation_rate": 1.5}, {"fill_tries": -1},
])
def test_invalid_parameters(params):
    with pytest.raises(ValueError):
        make(**params)
