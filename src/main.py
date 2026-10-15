"""Thin CLI: python -m src.main data/small.json baseline --seed 1 --param max_restart=5"""
import argparse
from random import Random
from typing import Any, Dict, List

from .algorithms import ALGORITHMS
from .core.constraints import Feasibility
from .core.initializer import random_state
from .core.io import load_problem, save_result
from .core.neighborhood import Neighborhood
from .core.objective import TotalShippingFee


def parse_params(items: List[str]) -> Dict[str, Any]:
    """['max_sideways=10', 'rate=0.5'] -> {'max_sideways': 10, 'rate': 0.5}"""
    params: Dict[str, Any] = {}
    for item in items:
        name, sep, raw = item.partition("=")
        if not sep or not name:
            raise ValueError(f"--param expects name=value, got {item!r}")
        for cast in (int, float):
            try:
                params[name] = cast(raw)
                break
            except ValueError:
                continue
        else:
            params[name] = raw
    return params


def main() -> None:
    parser = argparse.ArgumentParser(description="Ship cargo loading - local search")
    parser.add_argument("problem", help="path to a problem JSON (see data/)")
    parser.add_argument("algorithm", choices=sorted(ALGORITHMS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", help="write the full result JSON here (e.g. output/run1.json)")
    parser.add_argument("--param", action="append", default=[], metavar="NAME=VALUE",
                        help="algorithm parameter, repeatable (e.g. --param max_sideways=10)")
    args = parser.parse_args()

    problem = load_problem(args.problem)
    rng = Random(args.seed)
    initial = random_state(problem, rng)
    algorithm = ALGORITHMS[args.algorithm](Neighborhood(), TotalShippingFee(),
                                           **parse_params(args.param))
    result = algorithm.run(initial, rng)

    feasible = Feasibility().is_feasible(result.final_state)
    print(f"{result.algorithm}: {result.initial_value} -> {result.final_value} "
          f"in {result.iterations} iterations, {result.duration:.3f}s, feasible={feasible}")
    if args.out:
        save_result(result, args.out, seed=args.seed)


if __name__ == "__main__":
    main()
