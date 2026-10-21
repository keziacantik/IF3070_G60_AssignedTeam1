"""Genetic Algorithm figures, kept separate from plots.py (which it reuses, unchanged).

    python -m src.visualization.ga_plots data/medium.json "output/ga_medium_p*_i*_s*.json"

Per run (same names as the other algorithms):
  <run>_states.png     initial and final deck, value, weight, vehicles outside
  <run>_objective.png  population maximum and mean per generation
For all runs of one problem (<size> = problem file name, e.g. medium):
  ga_<size>_experiments.png  final value and duration when the number of iterations is
                             varied (population fixed) and when the population is varied
                             (iterations fixed); one dot per run, line = mean
  ga_<size>_summary.md       table per run and per configuration, for the report

Population size and iterations are read from extra.parameters, not from the file name.
"""
import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

from ..core.io import load_problem
from .__main__ import expand
from .plots import Run, load_run, plot_objective, plot_states


def _config(run: Run) -> Tuple[int, int]:
    params = run["extra"]["parameters"]
    return params["population_size"], params["iterations"]


def plot_experiments(runs: List[Run], path, control_population: int,
                     control_iterations: int) -> None:
    """Rows: iterations varied (population = control) / population varied (iterations =
    control). Columns: final value / duration."""
    groups = [
        ("Iterations", 1, 0, control_population,
         f"iterations varied (population = {control_population})"),
        ("Population size", 0, 1, control_iterations,
         f"population varied (iterations = {control_iterations})"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for row, (xlabel, varied, fixed, fixed_value, title) in enumerate(groups):
        chosen = [r for r in runs if _config(r)[fixed] == fixed_value]
        keys = sorted({_config(r)[varied] for r in chosen})
        for col, (field, ylabel) in enumerate((("final_value", "Final objective"),
                                               ("duration_seconds", "Duration (s)"))):
            ax = axes[row][col]
            for x, key in enumerate(keys):
                group = [r for r in chosen if _config(r)[varied] == key]
                ax.scatter([x] * len(group), [r[field] for r in group], color="C0", zorder=3,
                           label="run" if x == 0 else None)
            means = [_mean([r[field] for r in chosen if _config(r)[varied] == k]) for k in keys]
            ax.plot(range(len(keys)), means, color="C1", marker="_", markersize=20, label="mean")
            ax.set_xticks(range(len(keys)))
            ax.set_xticklabels([str(k) for k in keys])
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(title, fontsize=10)
            ax.grid(alpha=0.3)
            if keys:
                ax.legend(fontsize=8)
    fig.suptitle(f"Genetic Algorithm experiments ({len(runs)} runs)")
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def write_summary(runs: List[Run], path, title: str) -> None:
    lines = [f"# {title}", "",
             "| Population | Iterations | Seed | Initial value | Final value | "
             "Best generation | Duration (s) |",
             "|---:|---:|---:|---:|---:|---:|---:|"]
    for r in sorted(runs, key=lambda r: (_config(r), r["seed"])):
        population, iterations = _config(r)
        lines.append(f"| {population} | {iterations} | {r['seed']} | {r['initial_value']:g} | "
                     f"{r['final_value']:g} | {r['extra']['best_generation']} | "
                     f"{r['duration_seconds']:.2f} |")
    by_config: Dict[Tuple[int, int], List[Run]] = defaultdict(list)
    for r in runs:
        by_config[_config(r)].append(r)
    lines += ["", "Per configuration:", "",
              "| Population | Iterations | Runs | Mean final | Min | Max | Std dev | "
              "Mean duration (s) |",
              "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for (population, iterations), group in sorted(by_config.items()):
        finals = [r["final_value"] for r in group]
        mean = _mean(finals)
        std = _mean([(x - mean) ** 2 for x in finals]) ** 0.5
        lines.append(f"| {population} | {iterations} | {len(group)} | {mean:.1f} | "
                     f"{min(finals):g} | {max(finals):g} | {std:.1f} | "
                     f"{_mean([r['duration_seconds'] for r in group]):.2f} |")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _mean(values: List[float]) -> float:
    return sum(values) / len(values)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Plot Genetic Algorithm runs")
    parser.add_argument("problem", help="problem JSON the runs were made with (see data/)")
    parser.add_argument("runs", nargs="+", help="GA result JSON files (wildcards allowed)")
    parser.add_argument("--out-dir", default="output/plots")
    parser.add_argument("--control-population", type=int, default=50)
    parser.add_argument("--control-iterations", type=int, default=100)
    args = parser.parse_args(argv)

    problem = load_problem(args.problem)
    size = Path(args.problem).stem
    out_dir = Path(args.out_dir)
    runs = []
    for path in expand(args.runs):
        run = load_run(path)
        if run["algorithm"] != "ga":
            raise SystemExit(f"{path} is not a GA result (algorithm={run['algorithm']!r})")
        runs.append(run)
        stem = Path(path).stem
        plot_states(run, problem, out_dir / f"{stem}_states.png")
        plot_objective(run, out_dir / f"{stem}_objective.png")
        print(f"{path}: plotted")
    plot_experiments(runs, out_dir / f"ga_{size}_experiments.png",
                     args.control_population, args.control_iterations)
    write_summary(runs, out_dir / f"ga_{size}_summary.md", f"Genetic Algorithm, {size}")
    print(f"{out_dir / f'ga_{size}_experiments.png'}, {out_dir / f'ga_{size}_summary.md'}: written")


if __name__ == "__main__":
    main()
