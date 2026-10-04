"""Neighborhood: the only object an algorithm needs to move through the search space.

It composes Moves, filters by Feasibility and guarantees every state it returns is
feasible and different from the source.
"""
from random import Random
from typing import Iterable, List, Optional

from .constraints import Feasibility
from .moves import Move, RelocateMove, RotateMove, SwapMove
from .state import State


class Neighborhood:
    def __init__(self, moves: Iterable[Move] = None, feasibility: Feasibility = None,
                 max_attempts: int = 100):
        self.moves = tuple(moves if moves is not None
                           else (SwapMove(), RelocateMove(), RotateMove()))
        self.feasibility = feasibility or Feasibility()
        self.max_attempts = max_attempts

    def random_neighbor(self, state: State, rng: Random) -> Optional[State]:
        """One random feasible neighbour, or None if none was found in `max_attempts` draws.

        Used by Simulated Annealing, stochastic hill-climbing and GA mutation.
        """
        for _ in range(self.max_attempts):
            candidate = rng.choice(self.moves).sample(state, rng)
            if candidate is not None and self._acceptable(state, candidate):
                return candidate
        return None

    def all_neighbors(self, state: State, rng: Random = None,
                      limit: Optional[int] = None) -> List[State]:
        """Every distinct feasible neighbour (used by steepest-ascent / sideways HC).

        The full relocate neighbourhood is large; pass `limit` (and `rng`) to get a
        random subset of at most `limit` neighbours instead.
        """
        seen = set()
        result: List[State] = []
        for move in self.moves:
            for candidate in move.candidates(state):
                if candidate in seen or not self._acceptable(state, candidate):
                    continue
                seen.add(candidate)
                result.append(candidate)
        if limit is not None and len(result) > limit:
            result = (rng or Random()).sample(result, limit)
        return result

    def _acceptable(self, source: State, candidate: State) -> bool:
        return candidate != source and self.feasibility.is_feasible(candidate)
