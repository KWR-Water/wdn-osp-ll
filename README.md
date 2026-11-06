# wdn-osp-ll

Python library for Integrated Optimal Sensor Placement and Data-Driven Leak Localization for Water Distribution Networks.

## How to Cite

The work related to this library is currently under submission on Water Resources Management. 

In the meanwhile, please cite our Zenodo [![DOI](https://zenodo.org/badge/1448377757.svg)](https://doi.org/10.5281/zenodo.24013742).


## License 

This library is released under the [Creative Commons Attribution 4.0 International (CC BY 4.0)](./LICENSE) licence.

## Installation

### Core library and notebooks

Requires [uv](https://docs.astral.sh/uv/). Clone the repo, then:

```bash
uv sync --extra notebooks
```

This installs the `wdn_osp_ll` library in editable mode plus all dependencies needed to run the hydraulic analysis and experiment notebooks (`ipykernel`, `nbformat`, `epyt`).

### pygmo (required for optimization experiments)

`pygmo` is not managed by uv due to platform-specific installation requirements. Install it manually into the project environment after running `uv sync --extra pygmo-setup`:
