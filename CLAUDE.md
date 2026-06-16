# wdn-osp-ll

Research project and Python library for **Optimal Sensor Placement (OSP) in Water Distribution Networks (WDNs)** driven by actual leak localization (LL) performance, published alongside an academic paper.

## Project structure

```
wdn-osp-ll/
  src/wdn_osp_ll/        ← Python library (the reusable package)
    analysis/            ← hydraulic simulation data loading, graph analysis
    optimisation/        ← OSP problem definitions and algorithms
    plotting.py
  networks/              ← EPANET .inp files (committed to git)
  experiments/
    hyd_analysis/        ← hydraulic simulation configs (.yaml), one per (network + leak settings) combo
    *.ipynb              ← one notebook per experiment; each carries its own OSP config at the top
  .cache/                ← gitignored, local only; stores expensive precomputed data
    sim/                 ← cached EPANET simulation outputs, keyed by hyd_analysis config name
    opt/                 ← cached optimisation results, keyed by experiment
```

## Two-level config model

1. **Hydraulic simulation config** (`experiments/hyd_analysis/<name>.yaml`) — defines the EPANET network and leak simulation settings. Expensive to run; results cached in `.cache/sim/<name>/`.
2. **OSP experiment** (top cell of a notebook) — references a hyd_analysis config by name, defines the placement algorithm and its parameters. Results cached in `.cache/opt/<experiment_name>/`.

If a notebook points to an already-cached hyd_analysis, the EPANET runs are skipped automatically.

## Library: `wdn_osp_ll`

Installed as an editable uv project (`uv sync`). Import directly in notebooks — no path hacks needed.

Key modules:
- `wdn_osp_ll.analysis.wdn_data` — `SimulationData` class: loads or runs EPANET sims, owns `.cache/sim/`
- `wdn_osp_ll.optimisation.problems` — OSP problem definitions
- `wdn_osp_ll.plotting` — shared plotting utilities

## Setup

```bash
uv sync          # install deps + editable library
```

## Conventions

- Networks have no tanks (asserted on load).
- All pressures are stored in metres; flows in L/s.
- Simulation settings that affect outputs (emitter coefficients, pressure thresholds) are part of the hyd_analysis config and determine the cache key — changing them requires a new config name or `force_recompute: true`.
- Do not commit `.cache/`. Do not commit `.ipynb_checkpoints/`.
