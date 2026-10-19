"""Figures for one run (deck states, objective vs iteration, SA acceptance) and a
summary across runs. Reads only the result JSON written by `core.io.save_result`
plus the problem file it was run on; the algorithms themselves are not touched.

Deck convention (same as core.models): x grows to the right, y grows downward,
(x, y) is a vehicle's top-left cell.
"""
import json
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")  # files only, no window
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from ..core.io import state_from_dict  # noqa: E402
from ..core.problem import Problem  # noqa: E402
from ..core.state import State  # noqa: E402

Run = Dict[str, Any]


def load_run(path) -> Run:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def to_state(problem: Problem, data: Dict[str, Any]) -> State:
    """State from a saved state dict; a clear error if it belongs to another problem."""
    try:
        return state_from_dict(problem, data)
    except KeyError as missing:
        raise ValueError(f"vehicle {missing} is not in the problem file; "
                         "pass the problem the run was made with") from None


def _vehicle_color(index: int):
    return plt.get_cmap("tab20")(index % 20)  # same vehicle, same colour in every figure


def draw_deck(ax, problem: Problem, state: State, title: str) -> None:
    ship = problem.ship
    for i in state.placed_indices():
        x0, y0, x1, y1 = state.rect(i)
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=_vehicle_color(i),
                               edgecolor="black", linewidth=1.2))
        ax.text((x0 + x1) / 2, (y0 + y1) / 2, problem.vehicles[i].id, ha="center",
                va="center", fontsize=7, rotation=90 if y1 - y0 > x1 - x0 else 0)
    ax.set_xlim(0, ship.w)
    ax.set_ylim(ship.l, 0)  # y downward
    ax.set_aspect("equal")
    ax.set_xticks(range(0, ship.w + 1, max(1, ship.w // 10)))
    ax.set_yticks(range(0, ship.l + 1, max(1, ship.l // 10)))
    ax.set_xticks(range(ship.w + 1), minor=True)
    ax.set_yticks(range(ship.l + 1), minor=True)
    ax.grid(which="minor", color="lightgray", linewidth=0.5)
    ax.tick_params(labelsize=7)
    ax.set_title(title, fontsize=9)


def _state_summary(problem: Problem, state: State, value: float) -> str:
    outside = [problem.vehicles[i].id for i, p in enumerate(state.pos) if p is None]
    loaded = problem.n - len(outside)
    lines = [
        f"Value (total ShippingFee): {value}",
        f"Weight: {state.loaded_weight()} / {problem.ship.max_capacity}",
        f"Inside: {loaded} vehicles, outside: {len(outside)}",
        "Outside: " + (", ".join(outside) if outside else "-"),
    ]
    return "\n".join(textwrap.fill(line, 48, subsequent_indent="  ") for line in lines)


def plot_states(run: Run, problem: Problem, path) -> None:
    """Initial and final deck side by side, with value, weight and vehicles outside."""
    final_label = "Final state"
    if "best_restart" in run.get("extra", {}):  # random restart: final comes from another climb
        final_label += f" (best climb: restart {run['extra']['best_restart']})"
    states = [("Initial state", to_state(problem, run["initial_state"]), run["initial_value"]),
              (final_label, to_state(problem, run["final_state"]), run["final_value"])]
    ship = problem.ship
    deck_height, text_height = 6.5, 1.3
    deck_width = deck_height * ship.w / ship.l
    fig, axes = plt.subplots(2, 2, figsize=(max(9, 2 * deck_width + 2), deck_height + text_height + 0.6),
                             gridspec_kw={"height_ratios": [deck_height, text_height]})
    for col, (label, state, value) in enumerate(states):
        draw_deck(axes[0][col], problem, state, label)
        axes[1][col].axis("off")
        axes[1][col].text(0, 1, _state_summary(problem, state, value), va="top",
                          fontsize=8, family="monospace")
    fig.suptitle(_run_title(run))
    fig.tight_layout()
    _save(fig, path)


def plot_objective(run: Run, path) -> None:
    """Objective vs iteration. Iteration 0 is the initial state.

    GA results (extra has max_per_generation / mean_per_generation) are plotted as the
    population maximum and mean per generation instead of `history`.
    """
    extra = run.get("extra", {})
    fig, ax = plt.subplots(figsize=(9, 4.5))
    if "max_per_generation" in extra and "mean_per_generation" in extra:
        ax.plot(extra["max_per_generation"], label="population maximum")
        ax.plot(extra["mean_per_generation"], label="population mean")
        ax.set_xlabel("Generation")
    elif "iterations_per_restart" in extra:
        _plot_restarts(ax, run)
        ax.set_xlabel("Iteration (all climbs, one after another)")
    else:
        values = [run["initial_value"]] + run["history"]
        ax.plot(range(len(values)), values, linewidth=1,
                label="current value" if "best_history" in extra else "objective")
        if "best_history" in extra:
            best = [run["initial_value"]] + extra["best_history"]
            ax.plot(range(len(best)), best, linewidth=1.5, label="best so far")
        if extra.get("stuck_iterations"):
            stuck = extra["stuck_iterations"]
            ax.scatter(stuck, [values[k] for k in stuck], marker="x", color="red",
                       zorder=3, label=f"stuck detected ({len(stuck)})")
        ax.set_xlabel("Iteration")
    ax.set_ylabel("Objective (total ShippingFee)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title(_run_title(run), fontsize=10)
    fig.tight_layout()
    _save(fig, path)


def _plot_restarts(ax, run: Run) -> None:
    """Random restart: each climb is its own segment (a restart is a fresh start, not a
    downhill step), separated by dashed lines; the best climb is highlighted.

    A climb's start value is only known for the first one (the initial state), so the
    other segments begin at their first step. A climb with 0 steps is drawn as a dot.
    """
    extra, history = run["extra"], run["history"]
    best = extra["best_restart"]
    start = 0
    for k, n in enumerate(extra["iterations_per_restart"]):
        xs = list(range(start + 1, start + n + 1))
        ys = history[start:start + n]
        if k == 0:
            xs, ys = [0] + xs, [run["initial_value"]] + ys
        if not ys:
            xs, ys = [start], [extra["value_per_restart"][k]]
        is_best = k == best
        ax.plot(xs, ys, marker="." if len(ys) == 1 else None,
                color="C1" if is_best else "C0", linewidth=2 if is_best else 1,
                label=f"best climb (restart {best})" if is_best else ("climb" if k == 0 else None))
        start += n
        if k < len(extra["iterations_per_restart"]) - 1:
            ax.axvline(start + 0.5, color="gray", linestyle="--", linewidth=0.8,
                       label="restart" if k == 0 else None)


def plot_acceptance(run: Run, path) -> None:
    """Simulated Annealing: e^(dE/T) of every proposed worse move vs iteration."""
    points = run["extra"]["acceptance_probability"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    if points:
        iterations, probabilities = zip(*points)
        ax.scatter(iterations, probabilities, s=3, alpha=0.4)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("e^(ΔE/T)  (worse moves only)")
    ax.grid(alpha=0.3)
    extra = run["extra"]
    ax.set_title(f"{_run_title(run)}\nworse accepted {extra['accepted_worse_moves']}, "
                 f"rejected {extra['rejected_worse_moves']}, stuck {extra['stuck_count']}",
                 fontsize=10)
    fig.tight_layout()
    _save(fig, path)


def plot_summary(runs: List[Run], path, upper_bound: Optional[float] = None) -> None:
    """Final value and duration of every run, grouped by algorithm."""
    algorithms = list(dict.fromkeys(r["algorithm"] for r in runs))  # first-seen order
    fig, (ax_value, ax_time) = plt.subplots(1, 2, figsize=(12, 4.5))
    for pos, name in enumerate(algorithms):
        group = [r for r in runs if r["algorithm"] == name]
        values = [r["final_value"] for r in group]
        times = [r["duration_seconds"] for r in group]
        ax_value.scatter([pos] * len(values), values, color=f"C{pos}", zorder=3)
        ax_value.hlines(sum(values) / len(values), pos - 0.25, pos + 0.25, color=f"C{pos}")
        ax_time.scatter([pos] * len(times), times, color=f"C{pos}", zorder=3)
    if upper_bound is not None:
        ax_value.axhline(upper_bound, color="black", linestyle=":", linewidth=1,
                         label=f"upper bound {upper_bound:g}")
        ax_value.legend(fontsize=8)
    ax_value.set_ylabel("Final objective (dot = run, line = mean)")
    ax_time.set_ylabel("Duration (s)")
    ax_time.set_yscale("log")
    for ax in (ax_value, ax_time):
        ax.set_xticks(range(len(algorithms)))
        ax.set_xticklabels(algorithms, rotation=20, fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle(f"{len(runs)} runs")
    fig.tight_layout()
    _save(fig, path)


def _run_title(run: Run) -> str:
    return (f"{run['algorithm']}, seed {run['seed']}: {run['initial_value']} -> "
            f"{run['final_value']}, {run['iterations']} iterations, "
            f"{run['duration_seconds']:.2f} s")


def _save(fig, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
