from typing import Optional

import numpy as np

from ..base import SearchAreaDefinitionMethod
from .threshold_based import _check_bounds, _check_finite, _check_unit


class QuantileBasedSearchArea(SearchAreaDefinitionMethod):
    """
    Defines the search area by selecting nodes whose LSM score is at or above
    the q-th quantile of the per-scenario score distribution (axis=0).
    threshold (q) must be in [0, 1].
    """

    LABEL = "quantile"

    def __init__(
        self,
        threshold: Optional[float] = None,
        min_threshold: float = 0.0,
        max_threshold: float = 1.0,
    ):
        if threshold is not None:
            _check_finite(threshold, "threshold")
            _check_unit(threshold, "threshold")
        else:
            _check_bounds(min_threshold, max_threshold, bounded=True)
        self._threshold = threshold
        self._min = min_threshold
        self._max = max_threshold

    def _execute(self, reconstructed_lsm: np.ndarray, x: np.ndarray) -> np.ndarray:
        q = self._threshold if self._threshold is not None else x[0]
        return reconstructed_lsm >= np.quantile(reconstructed_lsm, q, axis=0)

    @property
    def nix(self) -> int:
        return 0


    def get_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        if self._threshold is not None:
            return (np.array([]), np.array([]))
        return (np.array([self._min]), np.array([self._max]))
