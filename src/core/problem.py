"""Problem instance: one ship plus the vehicles offered for loading."""
from typing import Iterable, Tuple

from .models import Orientation, Ship, Vehicle


class Problem:
    def __init__(self, ship: Ship, vehicles: Iterable[Vehicle]):
        self.ship = ship
        self.vehicles = tuple(vehicles)
        self.n = len(self.vehicles)

    def footprint(self, index: int, orientation: Orientation) -> Tuple[int, int]:
        """(width_x, height_y) in cells of vehicle `index` for `orientation`."""
        v = self.vehicles[index]
        if orientation == Orientation.HORIZONTAL:
            return (v.l, v.w)
        return (v.w, v.l)

    def fits_on_deck(self, index: int, orientation: Orientation) -> bool:
        fw, fh = self.footprint(index, orientation)
        return fw <= self.ship.w and fh <= self.ship.l
