from random import Random

import pytest

from src.algorithms.hill_climbing import SidewaysHillClimbing
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
    algo = SidewaysHillClimbing(Neighborhood(), TotalShippingFee(), neighbor_limit=200, **params)
    return algo.run(initial, rng)


def test_result_is_feasible_and_not_worse(problem):
    result = run(problem, 1, max_sideways=20)
    assert Feasibility().is_feasible(result.final_state)
    assert result.final_value >= result.initial_value


def test_history_never_decreases_and_matches_iterations(problem):
    result = run(problem, 2, max_sideways=20)
    assert result.iterations == len(result.history)
    assert all(b >= a for a, b in zip(result.history, result.history[1:]))


def test_same_seed_same_result(problem):
    a = run(problem, 3, max_sideways=20)
    b = run(problem, 3, max_sideways=20)
    assert a.final_state == b.final_state
    assert a.history == b.history


def test_zero_sideways_is_steepest_ascent(problem):
    result = run(problem, 1, max_sideways=0)
    assert result.extra["sideways_moves"] == 0
    assert result.extra["stop_reason"] in {"local_optimum", "max_sideways"}


def test_sideways_moves_bounded(problem):
    result = run(problem, 1, max_sideways=5)
    # moves are consecutive-limited; total can exceed 5 only through intervening improvements
    assert result.extra["sideways_moves"] <= result.iterations
