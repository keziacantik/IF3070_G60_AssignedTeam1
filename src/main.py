"""Thin CLI: python -m src.main data/small.json baseline --seed 1"""
import argparse
from random import Random

from .algorithms import ALGORITHMS
from .core.constraints import Feasibility
from .core.initializer import random_state
from .core.io import load_problem, save_result
from .core.neighborhood import Neighborhood
from .core.objective import TotalShippingFee


def main() -> None:
    parser = argparse.ArgumentParser(description="Ship cargo loading - local search")
    parser.add_argument("problem", help="path to a problem JSON (see data/)")
    parser.add_argument("algorithm", choices=sorted(ALGORITHMS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", help="write the full result JSON here (e.g. output/run1.json)")
    args = parser.parse_args()

    problem = load_problem(args.problem)
    rng = Random(args.seed)
    initial = random_state(problem, rng)
    algorithm = ALGORITHMS[args.algorithm](Neighborhood(), TotalShippingFee())
    result = algorithm.run(initial, rng)

    feasible = Feasibility().is_feasible(result.final_state)
    print(f"{result.algorithm}: {result.initial_value} -> {result.final_value} "
          f"in {result.iterations} iterations, {result.duration:.3f}s, feasible={feasible}")
    if args.out:
        save_result(result, args.out, seed=args.seed)


if __name__ == "__main__":
    main()
