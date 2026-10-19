# IF3070 Tugas Besar 1 - Ship Cargo Loading (Local Search)

Maximise total ShippingFee of vehicles loaded on a ship deck grid using local search
(Hill-climbing, Simulated Annealing, Genetic Algorithm). Python 3.10+, core uses stdlib only; visualization needs matplotlib.

## Setup & run
```
pip install pytest matplotlib
python -m pytest tests -q
python -m src.main data/medium.json baseline --seed 1 --out output/run1.json
python -m src.visualization data/medium.json output/run1.json
```

## Layout
```
src/core/        foundation (do not edit when adding an algorithm)
  models.py        Vehicle, Ship, Orientation
  problem.py       Problem (footprint rules)
  state.py         State: pos[i] = Placement(x, y, orientation) | None (outside); immutable
  constraints.py   NoOverlap, WithinBounds, WithinCapacity, Feasibility
  objective.py     Objective, TotalShippingFee
  moves.py         SwapMove, RelocateMove, RotateMove
  neighborhood.py  random_neighbor / all_neighbors (always feasible)
  initializer.py   random_state, repair (for GA crossover)
  search.py        LocalSearch (template), SearchResult, Recorder
  io.py            load_problem, save_result
src/algorithms/  one file per algorithm, registered with @register("name")
src/visualization/  plots of saved runs (matplotlib), reads output/*.json
data/            sample problems (small / medium / large)
tests/  docs/  output/
```
Conventions: (x, y) = top-left cell; HORIZONTAL footprint = (l, w), VERTICAL = (w, l).
Every state an algorithm sees is feasible; use `repair` after GA crossover.

## Adding an algorithm
Copy `src/algorithms/baseline.py`: subclass `LocalSearch`, implement `_search`, call
`recorder.record(value)` once per iteration, return `(final_state, extra_metrics_dict)`,
decorate with `@register("name")`, and import it in `src/algorithms/__init__.py`.
- Steepest ascent / sideways: `neighborhood.all_neighbors(state, rng, limit=...)`
  (full enumeration on `large.json` takes ~12 s, so use `limit`).
- SA / stochastic HC / GA mutation: `neighborhood.random_neighbor(state, rng)`.
- New objective: subclass `Objective`. New move: subclass `Move`, pass to `Neighborhood`.
- Always use the injected `rng` (never global `random`) so runs are reproducible.

## Status
- Hill-climbing, keempat varian selesai (`src/algorithms/hill_climbing.py`):
  `hc-steepest`, `hc-sideways` (`max_sideways`), `hc-stochastic` (`nmax`), `hc-random-restart` (`max_restart`).
  Contoh: `python -m src.main data/medium.json hc-random-restart --seed 1 --param max_restart=10 --out output/hc-random-restart_medium_s1.json`
  (`--param nama=nilai` boleh diulang untuk parameter algoritma apa pun).
- Simulated Annealing (`sa`): selesai. `python -m src.main data/medium.json sa --seed 1 --out output/sa_medium_s1.json`
- Visualisasi (`src/visualization/`): selesai untuk HC dan SA; plot GA siap jika GA menyimpan
  `extra.max_per_generation` dan `extra.mean_per_generation`.
  `python -m src.visualization data/medium.json "output/hc-*_medium_s*.json" "output/sa_medium_s*.json" --upper-bound 806`
  menulis ke `output/plots/`: `<run>_states.png` (geladak awal dan akhir, nilai, berat, kendaraan di luar),
  `<run>_objective.png`, `<run>_acceptance.png` (SA, e^(ΔE/T)), dan `summary.png` (semua run).
  Wajib memberikan file masalah yang sama dengan saat run.
- Genetic Algorithm: belum.
- Detail keputusan dan hasil: `docs/KEPUTUSAN.md`, `docs/EKSPERIMEN.md`.

## Pembagian tugas
_(to be filled in by the team)_
