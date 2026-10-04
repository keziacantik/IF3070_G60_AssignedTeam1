"""Random-improvement baseline. Doubles as the reference example of the LocalSearch
contract: copy this file's structure for a new algorithm."""
from random import Random
from typing import Any, Dict, Tuple

from ..core.neighborhood import Neighborhood
from ..core.objective import Objective
from ..core.search import LocalSearch, Recorder
from ..core.state import State
from . import register


@register("baseline")
class RandomImprove(LocalSearch):
    def __init__(self, neighborhood: Neighborhood, objective: Objective,
                 max_iterations: int = 200):
        super().__init__(neighborhood, objective)
        self.max_iterations = max_iterations

    def _search(self, initial: State, rng: Random,
                recorder: Recorder) -> Tuple[State, Dict[str, Any]]:
        current, value = initial, self.objective(initial)
        for _ in range(self.max_iterations):
            candidate = self.neighborhood.random_neighbor(current, rng)
            if candidate is not None:
                cand_value = self.objective(candidate)
                if cand_value > value:
                    current, value = candidate, cand_value
            recorder.record(value)
        return current, {}
