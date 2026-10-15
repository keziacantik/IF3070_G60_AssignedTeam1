from random import Random

import pytest

from src.algorithms import ALGORITHMS
from src.core.constraints import Feasibility
from src.core.initializer import random_state
from src.core.io import load_problem
from src.core.neighborhood import Neighborhood
from src.core.objective import TotalShippingFee
from src.main import parse_params

# small parameters keep the suite fast; defaults are exercised by the CLI runs
PARAMS = {
    "hc-steepest": {},
    "hc-sideways": {"max_sideways": 10},
    "hc-stochastic": {"nmax": 500},
    "hc-random-restart": {"max_restart": 3},
}


@pytest.fixture(scope="module")
def problem():
    return load_problem("data/medium.json")


def run(problem, name, seed, **params):
    rng = Random(seed)
    initial = random_state(problem, rng)
    algo = ALGORITHMS[name](Neighborhood(), TotalShippingFee(), **{**PARAMS[name], **params})
    return algo.run(initial, rng)


@pytest.mark.parametrize("name", sorted(PARAMS))
def test_result_is_feasible_and_not_worse(problem, name):
    result = run(problem, name, 1)
    assert Feasibility().is_feasible(result.final_state)
    assert result.final_value >= result.initial_value
    assert result.iterations == len(result.history)


@pytest.mark.parametrize("name", sorted(PARAMS))
def test_same_seed_same_result(problem, name):
    a = run(problem, name, 3)
    b = run(problem, name, 3)
    assert a.final_state == b.final_state
    assert a.history == b.history


@pytest.mark.parametrize("name", ["hc-steepest", "hc-sideways", "hc-stochastic"])
def test_history_never_decreases(problem, name):
    result = run(problem, name, 2)
    assert all(b >= a for a, b in zip(result.history, result.history[1:]))


def test_steepest_stops_at_true_local_optimum(problem):
    result = run(problem, "hc-steepest", 1)
    assert result.extra["stop_reason"] == "local_optimum"
    objective = TotalShippingFee()
    # brute force over the whole feasible neighbourhood: nothing is strictly better
    assert all(objective(n) <= result.final_value
               for n in Neighborhood().all_neighbors(result.final_state))


def test_steepest_takes_the_best_neighbor(problem):
    rng = Random(4)
    initial = random_state(problem, rng)
    result = ALGORITHMS["hc-steepest"](Neighborhood(), TotalShippingFee()).run(initial, rng)
    objective = TotalShippingFee()
    best_first_step = max(objective(n) for n in Neighborhood().all_neighbors(initial))
    assert result.history[0] == best_first_step


def test_zero_sideways_is_steepest_ascent(problem):
    result = run(problem, "hc-sideways", 1, max_sideways=0)
    assert result.extra["sideways_moves"] == 0
    steepest = run(problem, "hc-steepest", 1)
    assert result.final_state == steepest.final_state


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_sideways_never_worse_than_steepest(problem, seed):
    # both make identical steepest steps until steepest stops, then sideways continues
    assert run(problem, "hc-sideways", seed).final_value >= run(problem, "hc-steepest", seed).final_value


def test_sideways_respects_limit(problem):
    result = run(problem, "hc-sideways", 1, max_sideways=5)
    assert result.extra["stop_reason"] in {"max_sideways", "local_optimum"}
    assert result.extra["sideways_moves"] <= result.iterations


def test_stochastic_runs_exactly_nmax(problem):
    result = run(problem, "hc-stochastic", 1, nmax=300)
    assert result.iterations == 300
    assert result.extra["improvements"] <= 300


def test_random_restart_keeps_best_climb(problem):
    result = run(problem, "hc-random-restart", 1, max_restart=4)
    extra = result.extra
    assert len(extra["iterations_per_restart"]) == extra["restarts"] == 4
    assert sum(extra["iterations_per_restart"]) == result.iterations
    assert result.final_value == max(extra["value_per_restart"])
    assert extra["value_per_restart"][extra["best_restart"]] == result.final_value


def test_random_restart_first_climb_is_steepest(problem):
    restart = run(problem, "hc-random-restart", 2, max_restart=1)
    steepest = run(problem, "hc-steepest", 2)
    assert restart.final_state == steepest.final_state


def test_parse_params():
    assert parse_params(["max_sideways=10", "rate=0.5", "mode=fast"]) == {
        "max_sideways": 10, "rate": 0.5, "mode": "fast"}
    with pytest.raises(ValueError):
        parse_params(["oops"])
