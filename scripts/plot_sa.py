
import argparse
import glob
import json
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SEED_COLORS = [BLUE, ORANGE, AQUA]

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def style(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=10)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)


def load_runs(size):
    pattern = os.path.join(ROOT, "output", f"sa_{size}_s*.json")
    files = sorted(glob.glob(pattern),
                   key=lambda p: int(re.search(r"_s(\d+)\.json$", p).group(1)))
    runs = []
    for path in files:
        with open(path) as f:
            data = json.load(f)
        data["_file"] = os.path.basename(path)
        data["_k"] = int(re.search(r"_s(\d+)\.json$", path).group(1))
        runs.append(data)
    return runs


def plot_objective(run, size, out_dir):
    ex = run["extra"]
    hist, best = run["history"], ex["best_history"]
    it = range(1, len(hist) + 1)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(it, hist, color=BLUE, linewidth=1.0, label="Nilai state sekarang")
    ax.plot(it, best, color=ORANGE, linewidth=1.8, label="Nilai terbaik sejauh ini")
    stuck = ex.get("stuck_iterations", [])
    if stuck:
        lo = min(hist)
        ax.vlines(stuck, lo, lo + (max(best) - lo) * 0.06, color=MUTED, linewidth=1.2,
                  label=f"Stuck terdeteksi ({len(stuck)}x)")
    style(ax, f"SA {size}, seed {run['seed']}: objective function vs iterasi",
          "Iterasi", "Total ShippingFee (juta rupiah)")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    path = os.path.join(out_dir, f"sa_{size}_s{run['_k']}_objective.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_prob(run, size, out_dir):
    pts = run["extra"]["acceptance_probability"]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    if pts:
        xs, ys = zip(*pts)
        ax.scatter(xs, ys, s=4, color=BLUE, alpha=0.25, linewidths=0,
                   label="Tiap langkah lebih buruk")
        # rata-rata per kotak iterasi supaya tren menurun terlihat
        n_bins = 40
        span = max(xs) - min(xs) + 1
        width = max(1, span // n_bins)
        bins = {}
        for x, y in pts:
            bins.setdefault((x - min(xs)) // width, []).append(y)
        bx = [min(xs) + (b + 0.5) * width for b in sorted(bins)]
        by = [sum(bins[b]) / len(bins[b]) for b in sorted(bins)]
        ax.plot(bx, by, color=ORANGE, linewidth=2.0, label="Rata-rata per kotak iterasi")
    ax.set_ylim(-0.02, 1.02)
    style(ax, f"SA {size}, seed {run['seed']}: peluang menerima langkah lebih buruk, e^(ΔE/T)",
          "Iterasi", "e^(ΔE/T)")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    path = os.path.join(out_dir, f"sa_{size}_s{run['_k']}_prob.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_compare(runs, size, out_dir):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for i, run in enumerate(runs):
        best = run["extra"]["best_history"]
        ax.plot(range(1, len(best) + 1), best, color=SEED_COLORS[i % len(SEED_COLORS)],
                linewidth=1.6, label=f"Seed {run['seed']} (akhir {run['final_value']:g})")
    style(ax, f"SA {size}: nilai terbaik per iterasi, perbandingan antar seed",
          "Iterasi", "Nilai terbaik (juta rupiah)")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    path = os.path.join(out_dir, f"sa_{size}_compare.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def summary_table(runs, size, out_dir):
    head = ("| Run | Seed | Nilai awal | Nilai akhir | Iterasi | Durasi (s) | Stuck | "
            "Escape | Buruk diterima | Buruk ditolak |\n"
            "|---|---|---|---|---|---|---|---|---|---|")
    rows = []
    for k, r in enumerate(runs, 1):
        e = r["extra"]
        rows.append(f"| {k} | {r['seed']} | {r['initial_value']:g} | {r['final_value']:g} | "
                    f"{r['iterations']} | {r['duration_seconds']:.2f} | {e['stuck_count']} | "
                    f"{e['escapes']} | {e['accepted_worse_moves']} | {e['rejected_worse_moves']} |")
    finals = [r["final_value"] for r in runs]
    mean = sum(finals) / len(finals)
    std = (sum((x - mean) ** 2 for x in finals) / len(finals)) ** 0.5
    foot = (f"\nNilai akhir: rata-rata {mean:.1f}, min {min(finals):g}, max {max(finals):g}, "
            f"simpangan baku {std:.1f}.")
    text = f"SA, dataset {size}\n\n" + head + "\n" + "\n".join(rows) + "\n" + foot + "\n"
    path = os.path.join(out_dir, f"sa_{size}_summary.md")
    with open(path, "w") as f:
        f.write(text)
    return text


def main():
    parser = argparse.ArgumentParser(description="Plot hasil Simulated Annealing")
    parser.add_argument("size", nargs="?", default="medium", help="small | medium | large")
    args = parser.parse_args()

    runs = load_runs(args.size)
    if not runs:
        sys.exit(f"Tidak ada file output/sa_{args.size}_s*.json di {ROOT}. "
                 f"Jalankan dulu: python3 -m src.main data/{args.size}.json sa --seed 1 "
                 f"--out output/sa_{args.size}_s1.json")
    out_dir = os.path.join(ROOT, "output", "plots")
    os.makedirs(out_dir, exist_ok=True)
    for run in runs:
        print(plot_objective(run, args.size, out_dir))
        print(plot_prob(run, args.size, out_dir))
    print(plot_compare(runs, args.size, out_dir))
    print()
    print(summary_table(runs, args.size, out_dir))


if __name__ == "__main__":
    main()
