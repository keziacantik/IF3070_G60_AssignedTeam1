"""Simulated Annealing.

Accepts every non-worsening neighbour, and a worsening one (delta < 0) with
probability exp(delta / T). T cools geometrically (T <- T * cooling_rate), so the
search explores widely at first and behaves like hill-climbing near the end.

Spec-specific data is returned in `extra`:
  - acceptance_probability : [(iteration, min(1, e^(dE/T))), ...] for every iteration that
                             had a neighbour (plot this against iteration)
  - stuck_count            : worse neighbours that were REJECTED
  - accepted_worse_moves   : worse neighbours that were accepted (escapes from local optima)
  - no_neighbor_count      : iterations where Neighborhood found no feasible neighbour
  - best_value, final_temperature, parameters used
"""
import math
from random import Random
from typing import Any, Dict, Tuple

from ..core.neighborhood import Neighborhood
from ..core.objective import Objective
from ..core.search import LocalSearch, Recorder
from ..core.state import State
from . import register


@register("sa")
class SimulatedAnnealing(LocalSearch):
    def __init__(self, neighborhood: Neighborhood, objective: Objective,
                 initial_temperature: float = 100.0, cooling_rate: float = 0.999,
                 min_temperature: float = 1e-3, max_iterations: int = 20000):
        super().__init__(neighborhood, objective)
        if not 0 < cooling_rate < 1:
            raise ValueError("cooling_rate must be in (0, 1)")
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.min_temperature = min_temperature
        self.max_iterations = max_iterations

    def _search(self, initial: State, rng: Random,
                recorder: Recorder) -> Tuple[State, Dict[str, Any]]:
        current, value = initial, self.objective(initial)
        best, best_value = current, value
        temperature = self.initial_temperature

        probabilities = []
        stuck = accepted_worse = no_neighbor = 0
        iteration = 0

        while iteration < self.max_iterations and temperature > self.min_temperature:
            iteration += 1
            candidate = self.neighborhood.random_neighbor(current, rng)
            if candidate is None:
                no_neighbor += 1
            else:
                cand_value = self.objective(candidate)
                delta = cand_value - value          
                if delta >= 0:
                    probability, accept = 1.0, True  
                else:
                    probability = math.exp(delta / temperature)
                    accept = rng.random() < probability
                    if accept:
                        accepted_worse += 1
                    else:
                        stuck += 1
                probabilities.append((iteration, probability))
                if accept:
                    current, value = candidate, cand_value
                    if value > best_value:
                        best, best_value = current, value
            recorder.record(value)
            temperature *= self.cooling_rate

        extra = {
            "acceptance_probability": probabilities,
            "stuck_count": stuck,
            "accepted_worse_moves": accepted_worse,
            "no_neighbor_count": no_neighbor,
            "best_value": best_value,
            "final_temperature": temperature,
            "parameters": {
                "initial_temperature": self.initial_temperature,
                "cooling_rate": self.cooling_rate,
                "min_temperature": self.min_temperature,
                "max_iterations": self.max_iterations,
            },
        }
        return best, extra