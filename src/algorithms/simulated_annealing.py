"""Simulated Annealing.

Accepts every non-worsening neighbour, and a worsening one (delta < 0) with
probability exp(delta / T). T cools geometrically (T <- T * cooling_rate), so the
search explores widely at first and behaves like hill-climbing near the end.
The best state seen is returned (so final value = max of `best_history`).

"Stuck" (the spec asks for its frequency): the current value has not changed for
`stuck_window` consecutive iterations. Each such episode is counted once; it ends when
the value changes, either by an improvement or by accepting a worse move (an escape).

Spec-specific data is returned in `extra`:
  - acceptance_probability : [(iteration, e^(dE/T)), ...] for every proposed WORSE move
                             (dE < 0); plot this against iteration
  - stuck_count            : number of stuck episodes (definition above)
  - stuck_iterations       : iteration at which each stuck episode was detected
  - escapes                : stuck episodes left by accepting a worse move
  - accepted_worse_moves   : worse neighbours accepted
  - rejected_worse_moves   : worse neighbours rejected
  - no_neighbor_count      : iterations where Neighborhood found no feasible neighbour
  - best_history           : best value so far after each iteration (history = current value)
  - best_value, final_current_value, final_temperature, parameters used
"""
import math
from random import Random
from typing import Any, Dict, List, Tuple

from ..core.neighborhood import Neighborhood
from ..core.objective import Objective
from ..core.search import LocalSearch, Recorder
from ..core.state import State
from . import register


@register("sa")
class SimulatedAnnealing(LocalSearch):
    def __init__(self, neighborhood: Neighborhood, objective: Objective,
                 initial_temperature: float = 100.0, cooling_rate: float = 0.999,
                 min_temperature: float = 1e-3, max_iterations: int = 20000,
                 stuck_window: int = 100):
        super().__init__(neighborhood, objective)
        if not 0 < cooling_rate < 1:
            raise ValueError("cooling_rate must be in (0, 1)")
        if stuck_window < 1:
            raise ValueError("stuck_window must be >= 1")
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.min_temperature = min_temperature
        self.max_iterations = max_iterations
        self.stuck_window = stuck_window

    def _search(self, initial: State, rng: Random,
                recorder: Recorder) -> Tuple[State, Dict[str, Any]]:
        current, value = initial, self.objective(initial)
        best, best_value = current, value
        temperature = self.initial_temperature

        probabilities: List[Tuple[int, float]] = []
        best_history: List[float] = []
        stuck_iterations: List[int] = []
        accepted_worse = rejected_worse = no_neighbor = escapes = 0
        unchanged = 0          # consecutive iterations without a change in value
        stuck = False
        iteration = 0

        while iteration < self.max_iterations and temperature > self.min_temperature:
            iteration += 1
            previous = value
            candidate = self.neighborhood.random_neighbor(current, rng)
            if candidate is None:
                no_neighbor += 1
            else:
                cand_value = self.objective(candidate)
                delta = cand_value - value
                if delta >= 0:
                    accept = True
                else:
                    probability = math.exp(delta / temperature)
                    probabilities.append((iteration, probability))
                    accept = rng.random() < probability
                    if accept:
                        accepted_worse += 1
                    else:
                        rejected_worse += 1
                if accept:
                    current, value = candidate, cand_value
                    if value > best_value:
                        best, best_value = current, value

            if value == previous:
                unchanged += 1
                if unchanged == self.stuck_window:
                    stuck = True
                    stuck_iterations.append(iteration)
            else:
                if stuck and value < previous:
                    escapes += 1
                unchanged, stuck = 0, False

            recorder.record(value)
            best_history.append(best_value)
            temperature *= self.cooling_rate

        extra = {
            "acceptance_probability": probabilities,
            "stuck_count": len(stuck_iterations),
            "stuck_iterations": stuck_iterations,
            "escapes": escapes,
            "accepted_worse_moves": accepted_worse,
            "rejected_worse_moves": rejected_worse,
            "no_neighbor_count": no_neighbor,
            "best_history": best_history,
            "best_value": best_value,
            "final_current_value": value,
            "final_temperature": temperature,
            "parameters": {
                "initial_temperature": self.initial_temperature,
                "cooling_rate": self.cooling_rate,
                "min_temperature": self.min_temperature,
                "max_iterations": self.max_iterations,
                "stuck_window": self.stuck_window,
            },
        }
        return best, extra
