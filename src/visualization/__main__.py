"""CLI: python -m src.visualization data/medium.json output/hc-steepest_medium_s*.json

Per run it writes <name>_states.png and <name>_objective.png (plus <name>_acceptance.png
for Simulated Annealing); with more than one run also summary.png.
"""
import argparse
import glob
from pathlib import Path

from ..core.io import load_problem
from .plots import load_run, plot_acceptance, plot_objective, plot_states, plot_summary


def expand(patterns):
    """Expand wildcards ourselves so the command also works in PowerShell / cmd."""
    paths = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern)) if glob.has_magic(pattern) else [pattern]
        if not matches:
            raise SystemExit(f"no file matches {pattern!r}")
        paths.extend(matches)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot saved local-search runs")
    parser.add_argument("problem", help="problem JSON the runs were made with (see data/)")
    parser.add_argument("runs", nargs="+", help="result JSON files (wildcards allowed)")
    parser.add_argument("--out-dir", default="output/plots")
    parser.add_argument("--upper-bound", type=float,
                        help="known upper bound of the optimum, drawn on summary.png")
    parser.add_argument("--summary-name", default="summary.png")
    args = parser.parse_args()

    problem = load_problem(args.problem)
    out_dir = Path(args.out_dir)
    runs = []
    for path in expand(args.runs):
        run = load_run(path)
        runs.append(run)
        stem = Path(path).stem
        plot_states(run, problem, out_dir / f"{stem}_states.png")
        plot_objective(run, out_dir / f"{stem}_objective.png")
        if "acceptance_probability" in run.get("extra", {}):
            plot_acceptance(run, out_dir / f"{stem}_acceptance.png")
        print(f"{path}: plotted")
    if len(runs) > 1:
        plot_summary(runs, out_dir / args.summary_name, args.upper_bound)
        print(f"{out_dir / args.summary_name}: plotted")


if __name__ == "__main__":
    main()
