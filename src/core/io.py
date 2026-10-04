"""JSON in/out: load a problem, serialise states and results (into output/)."""
import json
from pathlib import Path
from typing import Any, Dict

from .models import Orientation, Ship, Vehicle
from .problem import Problem
from .search import SearchResult
from .state import Placement, State


def load_problem(path) -> Problem:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return problem_from_dict(data)


def problem_from_dict(data: Dict[str, Any]) -> Problem:
    ship = Ship(**data["ship"])
    vehicles = [Vehicle(**v) for v in data["vehicles"]]
    return Problem(ship, vehicles)


def state_to_dict(state: State) -> Dict[str, Any]:
    vehicles = state.problem.vehicles
    return {
        "loaded": [
            {"id": vehicles[i].id, "x": p.x, "y": p.y, "orientation": p.orientation.name}
            for i, p in enumerate(state.pos) if p is not None
        ],
        "outside": [vehicles[i].id for i, p in enumerate(state.pos) if p is None],
    }


def state_from_dict(problem: Problem, data: Dict[str, Any]) -> State:
    index = {v.id: i for i, v in enumerate(problem.vehicles)}
    pos = [None] * problem.n
    for item in data["loaded"]:
        pos[index[item["id"]]] = Placement(item["x"], item["y"], Orientation[item["orientation"]])
    return State(problem, pos)


def result_to_dict(result: SearchResult, seed: int = None) -> Dict[str, Any]:
    return {
        "algorithm": result.algorithm,
        "seed": seed,
        "initial_value": result.initial_value,
        "final_value": result.final_value,
        "iterations": result.iterations,
        "duration_seconds": result.duration,
        "initial_state": state_to_dict(result.initial_state),
        "final_state": state_to_dict(result.final_state),
        "history": result.history,
        "extra": result.extra,
    }


def save_result(result: SearchResult, path, seed: int = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result_to_dict(result, seed), indent=2), encoding="utf-8")
