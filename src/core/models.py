"""Immutable domain entities: Vehicle, Ship and Orientation.

Conventions (shared by every module):
  * A cell is addressed (x, y); x grows to the right, y grows downward.
  * A placement's (x, y) is the top-left cell of the vehicle.
  * HORIZONTAL footprint is (l, w) cells wide/high, VERTICAL is (w, l).
"""
from dataclasses import dataclass
from enum import IntEnum


class Orientation(IntEnum):
    HORIZONTAL = 0
    VERTICAL = 1

    def flipped(self) -> "Orientation":
        return Orientation(1 - self)


@dataclass(frozen=True)
class Vehicle:
    id: str
    w: int
    l: int
    fee: int          # ShippingFee (million rupiah)
    weight: int       # unit
    eta: int = 0      # kept from the spec; not used by any rule yet


@dataclass(frozen=True)
class Ship:
    w: int            # deck width  (cells)
    l: int            # deck length (cells)
    max_capacity: int # maximum total weight (unit)
