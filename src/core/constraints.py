"""Hard constraints. Add a rule by writing a new Constraint and passing it to Feasibility."""
from abc import ABC, abstractmethod
from typing import Iterable

from .state import Rect, State


class Constraint(ABC):
    @abstractmethod
    def is_satisfied(self, state: State) -> bool: ...


class WithinBounds(Constraint):
    """Every occupied cell lies inside the deck."""

    def is_satisfied(self, state: State) -> bool:
        ship = state.problem.ship
        for i in state.placed_indices():
            x0, y0, x1, y1 = state.rect(i)
            if x0 < 0 or y0 < 0 or x1 > ship.w or y1 > ship.l:
                return False
        return True


class NoOverlap(Constraint):
    """No deck cell is occupied by more than one vehicle."""

    @staticmethod
    def _overlap(a: Rect, b: Rect) -> bool:
        return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]

    def is_satisfied(self, state: State) -> bool:
        rects = [state.rect(i) for i in state.placed_indices()]
        for a in range(len(rects)):
            for b in range(a + 1, len(rects)):
                if self._overlap(rects[a], rects[b]):
                    return False
        return True


class WithinCapacity(Constraint):
    """Total loaded weight does not exceed the ship's MaxCapacity."""

    def is_satisfied(self, state: State) -> bool:
        return state.loaded_weight() <= state.problem.ship.max_capacity


class Feasibility:
    """Conjunction of constraints; the single place that answers 'is this state legal?'."""

    def __init__(self, constraints: Iterable[Constraint] = None):
        self.constraints = tuple(
            constraints if constraints is not None
            else (WithinBounds(), WithinCapacity(), NoOverlap())  # cheap checks first
        )

    def is_feasible(self, state: State) -> bool:
        return all(c.is_satisfied(state) for c in self.constraints)
