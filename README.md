# IF3070 Tugas Besar 1 - Ship Cargo Loading (Local Search)

Maximise total ShippingFee of vehicles loaded on a ship deck grid using local search
(Hill-climbing, Simulated Annealing, Genetic Algorithm). Python 3.10+, core uses stdlib only.

## Setup & run
```
pip install pytest
python -m pytest tests -q
python -m src.main data/medium.json baseline --seed 1 --out output/run1.json
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
src/visualization/  (to be added)
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

## Pembagian tugas
_(to be filled in by the team)_
