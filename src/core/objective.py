"""Objective functions (all are maximised). Add a new one by subclassing Objective."""
from abc import ABC, abstractmethod

from .state import State


class Objective(ABC):
    @abstractmethod
    def evaluate(self, state: State) -> float: ...

    def __call__(self, state: State) -> float:
        return self.evaluate(state)


class TotalShippingFee(Objective):
    """Sum of ShippingFee of every vehicle loaded on the ship (the spec's objective)."""

    def evaluate(self, state: State) -> float:
        vehicles = state.problem.vehicles
        return sum(vehicles[i].fee for i in state.placed_indices())
