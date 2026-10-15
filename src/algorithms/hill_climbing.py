"""Hill-climbing variants: steepest ascent, sideways move, stochastic and random restart.

Steepest ascent looks at the *whole* neighbourhood (every candidate of every Move). That
is affordable because a candidate is scored first (cheap) and only a strictly better one
is checked for feasibility (expensive), so the stop at a local optimum is exact.
"""
from random import Random
from typing import Any, Dict, List, Set, Tuple

from ..core.initializer import random_state
from ..core.neighborhood import Neighborhood
from ..core.objective import Objective
from ..core.search import LocalSearch, Recorder
from ..core.state import State
from . import register


class _HillClimbing(LocalSearch):
    def __init__(self, neighborhood: Neighborhood, objective: Objective,
                 max_iterations: int = 10_000):
        super().__init__(neighborhood, objective)
        self.max_iterations = max_iterations

    def _better_neighbors(self, state: State, value: float) -> Tuple[float, List[State]]:
        """(best value, every feasible neighbour with that value), considering only
        neighbours strictly better than `value`. Returns (value, []) if there are none."""
        best_value, best = value, []
        feasibility = self.neighborhood.feasibility
        for move in self.neighborhood.moves:
            for candidate in move.candidates(state):
                cand_value = self.objective(candidate)
                if cand_value <= value or cand_value < best_value:
                    continue
                if candidate == state or not feasibility.is_feasible(candidate):
                    continue
                if cand_value > best_value:
                    best_value, best = cand_value, []
                if candidate not in best:
                    best.append(candidate)
        return best_value, best

    def _climb(self, start: State, rng: Random,
               recorder: Recorder) -> Tuple[State, float, str]:
        """Steepest ascent from `start` until no neighbour is strictly better."""
        current, value = start, self.objective(start)
        for _ in range(self.max_iterations):
            best_value, best = self._better_neighbors(current, value)
            if not best:
                return current, value, "local_optimum"
            current, value = rng.choice(best), best_value
            recorder.record(value)
        return current, value, "max_iterations"


@register("hc-steepest")
class SteepestAscentHillClimbing(_HillClimbing):
    """Move to the best neighbour; stop when no neighbour is strictly better."""

    def _search(self, initial: State, rng: Random,
                recorder: Recorder) -> Tuple[State, Dict[str, Any]]:
        final, _, stop_reason = self._climb(initial, rng, recorder)
        return final, {"stop_reason": stop_reason}


@register("hc-sideways")
class SidewaysHillClimbing(_HillClimbing):
    """Steepest ascent that may also move to an equal-valued neighbour (sideways move).

    At most `max_sideways` sideways moves in a row; the counter resets whenever the value
    improves. States already visited on the current plateau are not revisited. With
    `max_sideways=0` this is plain steepest ascent.
    """

    def __init__(self, neighborhood: Neighborhood, objective: Objective,
                 max_sideways: int = 100, sideways_samples: int = 200,
                 max_iterations: int = 10_000):
        super().__init__(neighborhood, objective, max_iterations)
        self.max_sideways = max_sideways
        self.sideways_samples = sideways_samples

    def _sideways_neighbor(self, state: State, value: float, plateau: Set[State],
                           rng: Random):
        """A random unvisited neighbour with the same value, or None.

        Every equal-valued neighbour is a tied best neighbour, so a random draw among
        them is a faithful steepest-ascent tie-break; drawing avoids enumerating them all.
        """
        found: List[State] = []   # a list, not a set: keeps the draw order reproducible
        for _ in range(self.sideways_samples):
            neighbor = self.neighborhood.random_neighbor(state, rng)
            if (neighbor is not None and neighbor not in plateau and neighbor not in found
                    and self.objective(neighbor) == value):
                found.append(neighbor)
        return rng.choice(found) if found else None

    def _search(self, initial: State, rng: Random,
                recorder: Recorder) -> Tuple[State, Dict[str, Any]]:
        current, value = initial, self.objective(initial)
        sideways_in_row = 0
        sideways_total = 0
        plateau: Set[State] = {current}   # states already visited at this value
        stop_reason = "max_iterations"

        for _ in range(self.max_iterations):
            best_value, best = self._better_neighbors(current, value)
            if best:
                current, value = rng.choice(best), best_value
                sideways_in_row = 0
                plateau = {current}
            else:
                if sideways_in_row >= self.max_sideways:
                    stop_reason = "max_sideways"
                    break
                neighbor = self._sideways_neighbor(current, value, plateau, rng)
                if neighbor is None:
                    stop_reason = "local_optimum"
                    break
                current = neighbor
                plateau.add(current)
                sideways_in_row += 1
                sideways_total += 1
            recorder.record(value)

        return current, {
            "max_sideways": self.max_sideways,
            "sideways_moves": sideways_total,
            "stop_reason": stop_reason,
        }


@register("hc-stochastic")
class StochasticHillClimbing(_HillClimbing):
    """`nmax` times: draw one random neighbour and move to it only if strictly better."""

    def __init__(self, neighborhood: Neighborhood, objective: Objective,
                 nmax: int = 10_000):
        super().__init__(neighborhood, objective)
        self.nmax = nmax

    def _search(self, initial: State, rng: Random,
                recorder: Recorder) -> Tuple[State, Dict[str, Any]]:
        current, value = initial, self.objective(initial)
        improvements = 0
        for _ in range(self.nmax):
            neighbor = self.neighborhood.random_neighbor(current, rng)
            if neighbor is not None:
                neighbor_value = self.objective(neighbor)
                if neighbor_value > value:
                    current, value = neighbor, neighbor_value
                    improvements += 1
            recorder.record(value)
        return current, {"nmax": self.nmax, "improvements": improvements}


@register("hc-random-restart")
class RandomRestartHillClimbing(_HillClimbing):
    """Steepest ascent repeated from `max_restart` random initial states; keeps the best.

    The first climb starts from the given initial state, the rest from fresh random
    states. History is the concatenation of every climb (a sawtooth plot).
    """

    def __init__(self, neighborhood: Neighborhood, objective: Objective,
                 max_restart: int = 10, max_iterations: int = 10_000):
        super().__init__(neighborhood, objective, max_iterations)
        self.max_restart = max_restart

    def _search(self, initial: State, rng: Random,
                recorder: Recorder) -> Tuple[State, Dict[str, Any]]:
        best_state, best_value, best_restart = None, None, 0
        iterations_per_restart: List[int] = []
        value_per_restart: List[float] = []

        start = initial
        for restart in range(self.max_restart):
            if restart > 0:
                start = random_state(initial.problem, rng, self.neighborhood.feasibility)
            before = len(recorder.history)
            final, value, _ = self._climb(start, rng, recorder)
            iterations_per_restart.append(len(recorder.history) - before)
            value_per_restart.append(value)
            if best_value is None or value > best_value:
                best_state, best_value, best_restart = final, value, restart

        return best_state, {
            "max_restart": self.max_restart,
            "restarts": self.max_restart,
            "iterations_per_restart": iterations_per_restart,
            "value_per_restart": value_per_restart,
            "best_restart": best_restart,
        }
