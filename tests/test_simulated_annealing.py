import math
from random import Random

import pytest

from src.algorithms.simulated_annealing import SimulatedAnnealing
from src.core.constraints import Feasibility
from src.core.initializer import random_state
from src.core.io import load_problem
from src.core.neighborhood import Neighborhood
from src.core.objective import TotalShippingFee


@pytest.fixture(scope="module")
def problem():
    return load_problem("data/medium.json")


def run(problem, seed, **params):
    rng = Random(seed)
    initial = random_state(problem, rng)
    algo = SimulatedAnnealing(Neighborhood(), TotalShippingFee(), **params)
    return algo.run(initial, rng)


@pytest.fixture(scope="module")
def result(problem):
    return run(problem, 1)


def test_result_is_feasible_and_not_worse(result):
    assert Feasibility().is_feasible(result.final_state)
    assert result.final_value >= result.initial_value
    assert result.iterations == len(result.history)


def test_same_seed_same_result(problem):
    a = run(problem, 3, max_iterations=2000)
    b = run(problem, 3, max_iterations=2000)
    assert a.final_state == b.final_state
    assert a.history == b.history
    assert a.extra == b.extra


def test_returns_best_state_seen(result):
    best_history = result.extra["best_history"]
    assert len(best_history) == result.iterations
    assert all(b >= a for a, b in zip(best_history, best_history[1:]))
    assert result.final_value == result.extra["best_value"] == best_history[-1] == max(result.history)
    assert result.extra["final_current_value"] == result.history[-1]


def test_stops_on_temperature(result):
    p = result.extra["parameters"]
    expected = math.ceil(math.log(p["min_temperature"] / p["initial_temperature"])
                         / math.log(p["cooling_rate"]))
    assert result.iterations == expected
    assert result.extra["final_temperature"] <= p["min_temperature"]


def test_max_iterations_caps_run(problem):
    assert run(problem, 1, max_iterations=300).iterations == 300


def test_acceptance_probability_only_worse_moves(result):
    extra = result.extra
    probs = extra["acceptance_probability"]
    assert len(probs) == extra["accepted_worse_moves"] + extra["rejected_worse_moves"]
    assert all(0 <= p < 1 for _, p in probs)   # underflows to 0.0 when T is tiny
    iterations = [i for i, _ in probs]
    assert iterations == sorted(iterations) and iterations[-1] <= result.iterations


def test_stuck_episodes_match_history(result):
    """Recount stuck episodes from the recorded history and compare."""
    window = result.extra["parameters"]["stuck_window"]
    values = [result.initial_value] + result.history
    detected, run_length = [], 0
    for k in range(1, len(values)):
        run_length = run_length + 1 if values[k] == values[k - 1] else 0
        if run_length == window:
            detected.append(k)
    assert result.extra["stuck_iterations"] == detected
    assert result.extra["stuck_count"] == len(detected)
    assert 0 <= result.extra["escapes"] <= result.extra["stuck_count"]


def test_hot_start_accepts_more_worse_moves(result):
    probs = result.extra["acceptance_probability"]
    early = [p for i, p in probs if i <= 1000]
    late = [p for i, p in probs if i > result.iterations - 1000]
    assert sum(early) / len(early) > sum(late) / len(late)


def test_invalid_parameters():
    with pytest.raises(ValueError):
        SimulatedAnnealing(Neighborhood(), TotalShippingFee(), cooling_rate=1.0)
    with pytest.raises(ValueError):
        SimulatedAnnealing(Neighborhood(), TotalShippingFee(), stuck_window=0)
