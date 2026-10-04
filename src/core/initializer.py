"""Building feasible states: random initial state and repair of arbitrary placements."""
from random import Random
from typing import Optional, Sequence

from .constraints import Feasibility
from .models import Orientation
from .problem import Problem
from .state import Placement, State


def random_state(problem: Problem, rng: Random, feasibility: Feasibility = None,
                 tries_per_vehicle: int = 30) -> State:
    """Random feasible state: vehicles are tried in random order at random spots."""
    feasibility = feasibility or Feasibility()
    state = State(problem, [None] * problem.n)
    order = list(range(problem.n))
    rng.shuffle(order)
    for i in order:
        for _ in range(tries_per_vehicle):
            o = Orientation(rng.randrange(2))
            if not problem.fits_on_deck(i, o):
                continue
            fw, fh = problem.footprint(i, o)
            target = Placement(rng.randint(0, problem.ship.w - fw),
                               rng.randint(0, problem.ship.l - fh), o)
            candidate = state.with_placement(i, target)
            if feasibility.is_feasible(candidate):
                state = candidate
                break
    return state


def repair(problem: Problem, pos: Sequence[Optional[Placement]], rng: Random,
           feasibility: Feasibility = None) -> State:
    """Turn arbitrary placements (e.g. after GA crossover) into a feasible State.

    Placements are re-inserted in random order; any that would break a constraint is
    dropped (vehicle goes outside).
    """
    feasibility = feasibility or Feasibility()
    state = State(problem, [None] * problem.n)
    order = [i for i, p in enumerate(pos) if p is not None]
    rng.shuffle(order)
    for i in order:
        candidate = state.with_placement(i, pos[i])
        if feasibility.is_feasible(candidate):
            state = candidate
    return state
