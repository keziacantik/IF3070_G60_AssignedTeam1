"""Genetic Algorithm, following the course slides: each generation is built by
selection -> crossover -> mutation, and fully replaces the previous one (no elitism).

Chromosome : a State; gene i is the Placement of vehicle i, or None (outside the ship).
Fitness    : the objective (total ShippingFee of loaded vehicles).
Selection  : roulette wheel, probability proportional to fitness (uniform if all are 0).
Crossover  : single point on the vehicle index, giving two children. A child may break a
             constraint (overlap / capacity), so it is made feasible with `repair`, which
             leaves the clashing vehicles outside the ship.
Refill     : (our heuristic) parents' halves overlap a lot, so `repair` alone throws many
             vehicles out and children end up far worse than their parents (the best
             individual then always comes from the random initial population). After
             repair, every vehicle that is outside is tried at up to `fill_tries` random
             spots and loaded at the first feasible one. `fill_tries=0` gives the plain GA.
Mutation   : with probability `mutation_rate`, one random neighbour move of the spec
             (swap / relocate / rotate) via `Neighborhood.random_neighbor`.

The two parameters varied in the experiments are `population_size` and `iterations`
(number of generations). The best individual ever seen is returned as the final state.

`history` (from the Recorder) is the MAXIMUM fitness of each generation; it can go down
because there is no elitism. Spec-specific data is returned in `extra` (read by
src/visualization/plots.py and ga_plots.py):
  - max_per_generation     : maximum fitness of the population, index 0 = initial population
  - mean_per_generation    : average fitness of the population, index 0 = initial population
  - best_history           : best value seen so far after each generation
  - best_value, best_generation (0 = initial population)
  - parameters             : population_size, iterations, mutation_rate, fill_tries

Run (registers "ga" without touching src/algorithms/__init__.py, same flags as src.main):
    python -m src.algorithms.genetic data/medium.json ga --seed 1 \
        --param population_size=50 --param iterations=100 --out output/ga_medium_p50_i100_s1.json
"""
from random import Random
from typing import Any, Dict, List, Tuple

from ..core.initializer import random_state, repair
from ..core.models import Orientation
from ..core.neighborhood import Neighborhood
from ..core.objective import Objective
from ..core.search import LocalSearch, Recorder
from ..core.state import Placement, State
from . import register


@register("ga")
class GeneticAlgorithm(LocalSearch):
    def __init__(self, neighborhood: Neighborhood, objective: Objective,
                 population_size: int = 50, iterations: int = 100,
                 mutation_rate: float = 0.1, fill_tries: int = 10):
        super().__init__(neighborhood, objective)
        if population_size < 2:
            raise ValueError("population_size must be >= 2")
        if iterations < 1:
            raise ValueError("iterations must be >= 1")
        if not 0 <= mutation_rate <= 1:
            raise ValueError("mutation_rate must be in [0, 1]")
        if fill_tries < 0:
            raise ValueError("fill_tries must be >= 0")
        self.population_size = population_size
        self.iterations = iterations
        self.mutation_rate = mutation_rate
        self.fill_tries = fill_tries

    def _search(self, initial: State, rng: Random,
                recorder: Recorder) -> Tuple[State, Dict[str, Any]]:
        feasibility = self.neighborhood.feasibility
        population = [initial] + [random_state(initial.problem, rng, feasibility)
                                  for _ in range(self.population_size - 1)]
        fitness = [self.objective(s) for s in population]

        top = max(range(len(population)), key=fitness.__getitem__)
        best, best_value, best_generation = population[top], fitness[top], 0
        max_per_generation: List[float] = [fitness[top]]
        mean_per_generation: List[float] = [sum(fitness) / len(fitness)]
        best_history: List[float] = []

        for generation in range(1, self.iterations + 1):
            children: List[State] = []
            while len(children) < self.population_size:
                mother = self._select(population, fitness, rng)
                father = self._select(population, fitness, rng)
                for child in self._crossover(mother, father, rng):
                    children.append(self._mutate(child, rng))
            population = children[:self.population_size]
            fitness = [self.objective(s) for s in population]

            top = max(range(len(population)), key=fitness.__getitem__)
            if fitness[top] > best_value:
                best, best_value, best_generation = population[top], fitness[top], generation

            recorder.record(fitness[top])
            max_per_generation.append(fitness[top])
            mean_per_generation.append(sum(fitness) / len(fitness))
            best_history.append(best_value)

        extra = {
            "max_per_generation": max_per_generation,
            "mean_per_generation": mean_per_generation,
            "best_history": best_history,
            "best_value": best_value,
            "best_generation": best_generation,
            "parameters": {
                "population_size": self.population_size,
                "iterations": self.iterations,
                "mutation_rate": self.mutation_rate,
                "fill_tries": self.fill_tries,
            },
        }
        return best, extra

    @staticmethod
    def _select(population: List[State], fitness: List[float], rng: Random) -> State:
        """Roulette wheel: an individual is picked with probability fitness / total."""
        total = sum(fitness)
        if total <= 0:
            return rng.choice(population)
        pick = rng.uniform(0, total)
        cumulative = 0.0
        for state, value in zip(population, fitness):
            cumulative += value
            if pick <= cumulative:
                return state
        return population[-1]

    def _crossover(self, mother: State, father: State, rng: Random) -> Tuple[State, State]:
        """Single-point crossover on the vehicle index; both children are repaired, then refilled."""
        problem = mother.problem
        point = rng.randint(1, problem.n - 1) if problem.n > 1 else 0
        first = mother.pos[:point] + father.pos[point:]
        second = father.pos[:point] + mother.pos[point:]
        feasibility = self.neighborhood.feasibility
        return (self._fill(repair(problem, first, rng, feasibility), rng),
                self._fill(repair(problem, second, rng, feasibility), rng))

    def _fill(self, state: State, rng: Random) -> State:
        """Try to load every outside vehicle (random order) at a random feasible spot."""
        problem, feasibility = state.problem, self.neighborhood.feasibility
        outside = [i for i, p in enumerate(state.pos) if p is None]
        rng.shuffle(outside)
        for i in outside:
            for _ in range(self.fill_tries):
                o = Orientation(rng.randrange(2))
                if not problem.fits_on_deck(i, o):
                    continue
                fw, fh = problem.footprint(i, o)
                target = Placement(rng.randint(0, problem.ship.w - fw),
                                   rng.randint(0, problem.ship.l - fh), o)
                candidate = state.with_placement(i, target)
                if feasibility.is_feasible(candidate):
                    state = candidate
                    break
        return state

    def _mutate(self, state: State, rng: Random) -> State:
        """With probability mutation_rate, replace the child by a random neighbour."""
        if rng.random() >= self.mutation_rate:
            return state
        neighbour = self.neighborhood.random_neighbor(state, rng)
        return neighbour if neighbour is not None else state


if __name__ == "__main__":
    from ..main import main

    main()
