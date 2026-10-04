import time
from random import Random

import pytest

from src.algorithms import ALGORITHMS
from src.core.constraints import Feasibility, NoOverlap, WithinBounds, WithinCapacity
from src.core.initializer import random_state, repair
from src.core.io import load_problem, result_to_dict, state_from_dict, state_to_dict
from src.core.models import Orientation, Ship, Vehicle
from src.core.neighborhood import Neighborhood
from src.core.objective import TotalShippingFee
from src.core.problem import Problem
from src.core.state import Placement, State

H, V = Orientation.HORIZONTAL, Orientation.VERTICAL


@pytest.fixture
def problem():
    ship = Ship(w=10, l=20, max_capacity=100)
    return Problem(ship, [
        Vehicle("V01", w=2, l=4, fee=30, weight=8),
        Vehicle("V02", w=3, l=5, fee=60, weight=15),
        Vehicle("V03", w=1, l=2, fee=10, weight=3),
    ])


def test_footprint_orientation(problem):
    assert problem.footprint(0, H) == (4, 2)
    assert problem.footprint(0, V) == (2, 4)


def test_feasible_layout(problem):
    s = State(problem, [Placement(0, 0, H), None, Placement(5, 5, V)])
    assert Feasibility().is_feasible(s)


def test_overlap_detected(problem):
    s = State(problem, [Placement(0, 0, H), None, Placement(1, 0, H)])
    assert not NoOverlap().is_satisfied(s)
    assert not Feasibility().is_feasible(s)


def test_touching_is_not_overlap(problem):
    s = State(problem, [Placement(0, 0, H), None, Placement(4, 0, H)])  # V01 spans x 0..3
    assert NoOverlap().is_satisfied(s)


def test_out_of_bounds_detected(problem):
    s = State(problem, [Placement(8, 0, H), None, None])  # needs x 8..11 on a width-10 deck
    assert not WithinBounds().is_satisfied(s)


def test_overweight_detected(problem):
    tight = Problem(Ship(10, 20, 10), problem.vehicles)
    s = State(tight, [Placement(0, 0, H), None, Placement(5, 5, V)])  # 8 + 3 > 10
    assert not WithinCapacity().is_satisfied(s)


def test_objective_counts_only_loaded(problem):
    s = State(problem, [Placement(0, 0, H), None, Placement(5, 5, V)])
    assert TotalShippingFee()(s) == 40


def test_state_is_immutable_and_hashable(problem):
    s = State(problem, [None, None, None])
    t = s.with_placement(0, Placement(0, 0, H))
    assert s.pos == (None, None, None) and t != s and len({s, t}) == 2


@pytest.mark.parametrize("name", ["small", "medium", "large"])
def test_random_state_always_feasible(name):
    p = load_problem(f"data/{name}.json")
    for seed in range(20):
        assert Feasibility().is_feasible(random_state(p, Random(seed)))


def test_repair_returns_feasible_state(problem):
    clashing = [Placement(0, 0, H), Placement(0, 0, H), Placement(0, 0, H)]
    fixed = repair(problem, clashing, Random(1))
    assert Feasibility().is_feasible(fixed)
    assert len(list(fixed.placed_indices())) == 1


@pytest.mark.parametrize("name", ["small", "medium"])
def test_neighbors_are_feasible_and_different(name):
    p = load_problem(f"data/{name}.json")
    nb = Neighborhood()
    feas = Feasibility()
    for seed in range(10):
        s = random_state(p, Random(seed))
        rng = Random(seed)
        for _ in range(30):
            n = nb.random_neighbor(s, rng)
            assert n is not None and n != s and feas.is_feasible(n)
        everything = nb.all_neighbors(s)
        assert len(everything) == len(set(everything))
        assert all(x != s and feas.is_feasible(x) for x in everything)


def test_all_neighbors_limit(problem):
    nb = Neighborhood()
    s = random_state(problem, Random(3))
    assert len(nb.all_neighbors(s, Random(0), limit=5)) == 5


def test_all_neighbors_fast_enough_on_large():
    p = load_problem("data/large.json")
    s = random_state(p, Random(0))
    t = time.perf_counter()
    Neighborhood().all_neighbors(s, Random(0), limit=200)
    assert time.perf_counter() - t < 60  # generous bound; report the real number


def test_io_roundtrip(problem):
    s = State(problem, [Placement(0, 0, H), None, Placement(5, 5, V)])
    assert state_from_dict(problem, state_to_dict(s)) == s


def test_baseline_contract_and_determinism():
    p = load_problem("data/medium.json")

    def run(seed):
        rng = Random(seed)
        algo = ALGORITHMS["baseline"](Neighborhood(), TotalShippingFee())
        return algo.run(random_state(p, rng), rng)

    a, b = run(7), run(7)
    assert a.final_state == b.final_state and a.history == b.history   # reproducible
    assert a.iterations == len(a.history)
    assert a.final_value >= a.initial_value
    assert Feasibility().is_feasible(a.final_state)
    assert result_to_dict(a, seed=7)["algorithm"] == "baseline"
