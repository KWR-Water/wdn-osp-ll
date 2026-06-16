# wdn-osp-ll

Integrated Optimal Sensor Placement and data-driven Leak Localization for Water Distribution Networks.

## Installation

### Core library and notebooks

Requires [uv](https://docs.astral.sh/uv/). Clone the repo, then:

```bash
uv sync --extra notebooks
```

This installs the `wdn_osp_ll` library in editable mode plus all dependencies needed to run the hydraulic analysis and experiment notebooks (`ipykernel`, `nbformat`, `epyt`).

### pygmo (required for optimization experiments)

`pygmo` is not managed by uv due to platform-specific installation requirements. Install it manually into the project environment after running `uv sync --extra pygmo-setup`:
