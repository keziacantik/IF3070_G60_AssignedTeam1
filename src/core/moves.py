"""Neighbour moves. Each Move proposes states; it does NOT check feasibility
(that is the Neighborhood's job), so a move stays a small, single-purpose class.

To add a move: subclass Move and pass an instance to Neighborhood.
"""
from abc import ABC, abstractmethod
from random import Random
from typing import Iterator, Optional

from .models import Orientation
from .state import Placement, State


class Move(ABC):
    @abstractmethod
    def candidates(self, state: State) -> Iterator[State]:
        """Every state this move can reach from `state` (may include infeasible ones)."""

    @abstractmethod
    def sample(self, state: State, rng: Random) -> Optional[State]:
        """One random state reachable by this move, or None if the draw is not applicable."""


class SwapMove(Move):
    """Exchange the placements of two vehicles (inside or outside the ship)."""

    def candidates(self, state: State) -> Iterator[State]:
        n = state.problem.n
        for i in range(n):
            for j in range(i + 1, n):
                if state.pos[i] is None and state.pos[j] is None:
                    continue
                yield self._swap(state, i, j)

    def sample(self, state: State, rng: Random) -> Optional[State]:
        n = state.problem.n
        if n < 2:
            return None
        i, j = rng.sample(range(n), 2)
        return self._swap(state, i, j)

    @staticmethod
    def _swap(state: State, i: int, j: int) -> State:
        pos = list(state.pos)
        pos[i], pos[j] = pos[j], pos[i]
        return State(state.problem, pos)


class RelocateMove(Move):
    """Put one vehicle at a different coordinate, or take it off the ship.

    A vehicle that is outside is loaded (in either orientation); a vehicle that is
    inside keeps its orientation and changes position (or is unloaded).
    """

    def __init__(self, unload_probability: float = 0.15):
        self.unload_probability = unload_probability

    def candidates(self, state: State) -> Iterator[State]:
        problem = state.problem
        for i in range(problem.n):
            current = state.pos[i]
            orientations = (current.orientation,) if current else tuple(Orientation)
            for o in orientations:
                if not problem.fits_on_deck(i, o):
                    continue
                fw, fh = problem.footprint(i, o)
                for x in range(problem.ship.w - fw + 1):
                    for y in range(problem.ship.l - fh + 1):
                        target = Placement(x, y, o)
                        if target != current:
                            yield state.with_placement(i, target)
            if current is not None:
                yield state.with_placement(i, None)

    def sample(self, state: State, rng: Random) -> Optional[State]:
        problem = state.problem
        i = rng.randrange(problem.n)
        current = state.pos[i]
        if current is not None and rng.random() < self.unload_probability:
            return state.with_placement(i, None)
        o = current.orientation if current else Orientation(rng.randrange(2))
        if not problem.fits_on_deck(i, o):
            return None
        fw, fh = problem.footprint(i, o)
        x = rng.randint(0, problem.ship.w - fw)
        y = rng.randint(0, problem.ship.l - fh)
        return state.with_placement(i, Placement(x, y, o))


class RotateMove(Move):
    """Flip the orientation of a vehicle that is on the ship (top-left cell stays)."""

    def candidates(self, state: State) -> Iterator[State]:
        for i in state.placed_indices():
            rotated = self._rotate(state, i)
            if rotated is not None:
                yield rotated

    def sample(self, state: State, rng: Random) -> Optional[State]:
        placed = list(state.placed_indices())
        if not placed:
            return None
        return self._rotate(state, rng.choice(placed))

    @staticmethod
    def _rotate(state: State, i: int) -> Optional[State]:
        p = state.pos[i]
        flipped = p.orientation.flipped()
        if not state.problem.fits_on_deck(i, flipped):
            return None
        return state.with_placement(i, Placement(p.x, p.y, flipped))
