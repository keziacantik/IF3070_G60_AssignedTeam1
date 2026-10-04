"""The contract every local-search algorithm plugs into (Template Method).

To add an algorithm, subclass LocalSearch and implement `_search` only. Timing,
result assembly and history bookkeeping are handled here, so every algorithm
reports the same data the spec asks for.
"""
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from random import Random
from typing import Any, Dict, List, Tuple

from .neighborhood import Neighborhood
from .objective import Objective
from .state import State


class Recorder:
    """Collects the objective value after each iteration (for the objective-vs-iteration plot)."""

    def __init__(self):
        self.history: List[float] = []

    def record(self, value: float) -> None:
        self.history.append(value)


@dataclass
class SearchResult:
    algorithm: str
    initial_state: State
    final_state: State
    initial_value: float
    final_value: float
    history: List[float]          # history[k] = value after iteration k+1
    iterations: int               # == len(history)
    duration: float               # seconds
    extra: Dict[str, Any] = field(default_factory=dict)  # algorithm-specific metrics


class LocalSearch(ABC):
    name = "local-search"

    def __init__(self, neighborhood: Neighborhood, objective: Objective):
        self.neighborhood = neighborhood
        self.objective = objective

    def run(self, initial: State, rng: Random) -> SearchResult:
        recorder = Recorder()
        started = time.perf_counter()
        final, extra = self._search(initial, rng, recorder)
        duration = time.perf_counter() - started
        return SearchResult(
            algorithm=self.name,
            initial_state=initial,
            final_state=final,
            initial_value=self.objective(initial),
            final_value=self.objective(final),
            history=recorder.history,
            iterations=len(recorder.history),
            duration=duration,
            extra=extra,
        )

    @abstractmethod
    def _search(self, initial: State, rng: Random,
                recorder: Recorder) -> Tuple[State, Dict[str, Any]]:
        """Run the search. Call `recorder.record(value)` once per iteration.

        Return (final_state, extra_metrics).
        """
