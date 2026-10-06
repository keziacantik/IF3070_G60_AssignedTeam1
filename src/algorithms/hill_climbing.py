"""Steepest-ascent hill-climbing with sideways moves.

Each iteration looks at a sample of `neighbor_limit` neighbours of the current state and
moves to the best one.
A strictly better neighbour is always taken. A neighbour with the same value (a sideways
move) is taken too, but at most `max_sideways` times in a row; the counter resets whenever
the value improves. With `max_sideways=0` this is plain steepest-ascent.
"""
from random import Random
from typing import Any, Dict, List, Set, Tuple

from ..core.neighborhood import Neighborhood
from ..core.objective import Objective
from ..core.search import LocalSearch, Recorder
from ..core.state import State
from . import register


@register("hill-climbing")
class SidewaysHillClimbing(LocalSearch):
    def __init__(self, neighborhood: Neighborhood, objective: Objective,
                 max_sideways: int = 100, neighbor_limit: int = 200,
                 max_iterations: int = 10_000):
        super().__init__(neighborhood, objective)
        self.max_sideways = max_sideways
        self.neighbor_limit = neighbor_limit
        self.max_iterations = max_iterations

    def _sample_neighbors(self, state: State, rng: Random) -> List[State]:
        """Up to `neighbor_limit` distinct random feasible neighbours.

        `all_neighbors` enumerates the whole neighbourhood before sampling (about 12 s per
        call on large.json), so the candidate set is drawn with `random_neighbor` instead.
        """
        found: Set[State] = set()
        for _ in range(self.neighbor_limit):
            neighbor = self.neighborhood.random_neighbor(state, rng)
            if neighbor is not None:
                found.add(neighbor)
        return list(found)

    def _search(self, initial: State, rng: Random,
                recorder: Recorder) -> Tuple[State, Dict[str, Any]]:
        current, value = initial, self.objective(initial)
        sideways_in_row = 0          # consecutive sideways moves since the last improvement
        sideways_total = 0
        plateau: Set[State] = {current}   # states already visited at this value
        stop_reason = "max_iterations"

        for _ in range(self.max_iterations):
            neighbors = self._sample_neighbors(current, rng)
            if not neighbors:
                stop_reason = "no_neighbors"
                break

            scored = [(self.objective(n), n) for n in neighbors]
            best_value = max(v for v, _ in scored)

            if best_value < value:
                stop_reason = "local_optimum"
                break

            if best_value > value:
                best = [n for v, n in scored if v == best_value]
                current, value = rng.choice(best), best_value
                sideways_in_row = 0
                plateau = {current}
            else:
                if sideways_in_row >= self.max_sideways:
                    stop_reason = "max_sideways"
                    break
                fresh = [n for v, n in scored if v == best_value and n not in plateau]
                if not fresh:
                    stop_reason = "local_optimum"
                    break
                current = rng.choice(fresh)
                plateau.add(current)
                sideways_in_row += 1
                sideways_total += 1

            recorder.record(value)

        return current, {
            "max_sideways": self.max_sideways,
            "neighbor_limit": self.neighbor_limit,
            "sideways_moves": sideways_total,
            "stop_reason": stop_reason,
        }
