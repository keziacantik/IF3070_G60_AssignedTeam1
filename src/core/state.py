"""State: an immutable assignment of every vehicle to a placement or to 'outside'."""
from typing import Iterator, NamedTuple, Optional, Sequence, Tuple

from .models import Orientation
from .problem import Problem


class Placement(NamedTuple):
    x: int
    y: int
    orientation: Orientation


Rect = Tuple[int, int, int, int]  # (x_left, y_top, x_right, y_bottom), right/bottom exclusive


class State:
    """pos[i] is the Placement of vehicle i, or None if it stays outside the ship.

    A State is pure data. Whether it is feasible is decided by `Feasibility`
    and how good it is by an `Objective`; neither lives here.
    """

    __slots__ = ("problem", "pos")

    def __init__(self, problem: Problem, pos: Sequence[Optional[Placement]]):
        self.problem = problem
        self.pos: Tuple[Optional[Placement], ...] = tuple(pos)

    def rect(self, index: int) -> Optional[Rect]:
        p = self.pos[index]
        if p is None:
            return None
        fw, fh = self.problem.footprint(index, p.orientation)
        return (p.x, p.y, p.x + fw, p.y + fh)

    def placed_indices(self) -> Iterator[int]:
        return (i for i, p in enumerate(self.pos) if p is not None)

    def loaded_weight(self) -> int:
        return sum(self.problem.vehicles[i].weight for i in self.placed_indices())

    def with_placement(self, index: int, placement: Optional[Placement]) -> "State":
        pos = list(self.pos)
        pos[index] = placement
        return State(self.problem, pos)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, State) and self.pos == other.pos

    def __hash__(self) -> int:
        return hash(self.pos)

    def __repr__(self) -> str:
        return f"State({self.pos})"
