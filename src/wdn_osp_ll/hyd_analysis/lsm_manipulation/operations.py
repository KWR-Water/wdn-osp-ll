"""Stateless operations on leak signature matrices.

Each operation works on any :class:`LeakSignatureMatrix` subclass: it reads the
labels it needs and either returns a plain array (``observe``) or a new matrix
of the same subclass via ``_replace`` (``scale_*``). None of them mutate the
input.
"""

import numpy as np

from .leak_signature_matrices import LeakSignatureMatrix


def observe(
        lsm: LeakSignatureMatrix,
        sensor_ids: list[str] = [],
        sensor_idxs: list[int] | np.ndarray = [],    
    ) -> np.ndarray:
    """Mask the matrix down to a sensor set: same shape as ``lsm.data`` but with
    every non-sensor row set to NaN.

    Returns a plain array (not a new LSM): an observation is a view of the data
    through the sensors, not a different signature matrix.
    """
    if len(sensor_ids)>0:
        observed = lsm.data.copy()
        sensors = set(sensor_ids)
        keep = np.array([node_id in sensors for node_id in lsm.node_ids])
        observed[~keep, :] = np.nan
    elif len(sensor_idxs)>0:
        sensor_idxs = np.array(sensor_idxs)
        observed = np.full_like(lsm.data, fill_value=np.nan)
        observed[sensor_idxs, :] = lsm.data[sensor_idxs, :]
    else:
        raise ValueError("You didn't specify which sensor to use. Specify either the IDs or the indexes")
    
    return observed

def scale_nodes_by_scenario_mean(lsm: LeakSignatureMatrix) -> LeakSignatureMatrix:
    """Normalise each node (row) by its own mean across the scenarios.

    One factor per node: the mean of that row over all columns. Returns a new
    matrix of the same subclass; the removed per-node means are folded into
    ``scale_factors`` so the result can still be inverted back to a pressure
    field.

    A row whose mean is exactly zero carries no signal — e.g. reservoirs, whose
    pressure never changes, so their sensitivity is zero in every scenario. Such
    rows are scaled by 1 (left untouched) instead of dividing by zero.
    """
    means = lsm.data.mean(axis=1, keepdims=True)  # (n_nodes, 1)
    means = np.where(means == 0, 1.0, means)
    return lsm._scale(means)
