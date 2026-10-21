import json
from random import Random

import pytest

pytest.importorskip("matplotlib")

from src.algorithms import ALGORITHMS  # noqa: E402
from src.core.initializer import random_state  # noqa: E402
from src.core.io import load_problem, result_to_dict  # noqa: E402
from src.core.neighborhood import Neighborhood  # noqa: E402
from src.core.objective import TotalShippingFee  # noqa: E402
from src.visualization import plots  # noqa: E402
from src.visualization.__main__ import expand  # noqa: E402


@pytest.fixture(scope="module")
def problem():
    return load_problem("data/medium.json")


def saved_run(problem, name, seed=1, **params):
    """A result dict exactly as save_result writes it (JSON round trip)."""
    rng = Random(seed)
    initial = random_state(problem, rng)
    result = ALGORITHMS[name](Neighborhood(), TotalShippingFee(), **params).run(initial, rng)
    return json.loads(json.dumps(result_to_dict(result, seed)))


def test_states_and_objective_for_hill_climbing(problem, tmp_path):
    run = saved_run(problem, "hc-steepest")
    plots.plot_states(run, problem, tmp_path / "states.png")
    plots.plot_objective(run, tmp_path / "objective.png")
    assert (tmp_path / "states.png").stat().st_size > 0
    assert (tmp_path / "objective.png").stat().st_size > 0


def test_random_restart_objective(problem, tmp_path):
    run = saved_run(problem, "hc-random-restart", max_restart=3)
    plots.plot_states(run, problem, tmp_path / "states.png")
    plots.plot_objective(run, tmp_path / "objective.png")
    assert (tmp_path / "objective.png").exists()


def test_simulated_annealing_acceptance(problem, tmp_path):
    run = saved_run(problem, "sa", cooling_rate=0.99, stuck_window=20)
    plots.plot_objective(run, tmp_path / "objective.png")
    plots.plot_acceptance(run, tmp_path / "acceptance.png")
    assert (tmp_path / "acceptance.png").exists()


def test_ga_style_result_uses_population_curves(problem, tmp_path):
    run = saved_run(problem, "baseline", max_iterations=20)
    run["extra"] = {"max_per_generation": [600, 650, 700],
                    "mean_per_generation": [500, 560, 640]}
    plots.plot_objective(run, tmp_path / "objective.png")
    assert (tmp_path / "objective.png").exists()


def test_summary(problem, tmp_path):
    runs = [saved_run(problem, "baseline", seed=s, max_iterations=20) for s in (1, 2)]
    plots.plot_summary(runs, tmp_path / "summary.png", upper_bound=806)
    assert (tmp_path / "summary.png").exists()


def test_wrong_problem_file_is_reported(problem):
    run = saved_run(load_problem("data/large.json"), "baseline", max_iterations=5)
    with pytest.raises(ValueError, match="problem"):
        plots.to_state(problem, run["final_state"])


def test_expand_handles_wildcards(tmp_path):
    for name in ("a_s1.json", "a_s2.json"):
        (tmp_path / name).write_text("{}")
    assert len(expand([str(tmp_path / "a_s*.json")])) == 2
    with pytest.raises(SystemExit):
        expand([str(tmp_path / "missing_*.json")])
