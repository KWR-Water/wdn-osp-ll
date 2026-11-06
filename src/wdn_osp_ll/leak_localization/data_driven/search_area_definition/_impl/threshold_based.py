from typing import Optional

import numpy as np

from ..base import SearchAreaDefinitionMethod


def _check_finite(value: float, name: str) -> None:
    if np.isnan(value) or np.isinf(value):
        raise ValueError(f"{name} must be finite, got {value!r}")


def _check_unit(value: float, name: str) -> None:
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"{name} must be in [0, 1], got {value!r}")


def _check_bounds(lo: float, hi: float, bounded: bool) -> None:
    for val, name in ((lo, "min_threshold"), (hi, "max_threshold")):
        _check_finite(val, name)
        if bounded:
            _check_unit(val, name)
    if lo >= hi:
        raise ValueError(
            f"min_threshold ({lo}) must be strictly less than max_threshold ({hi})"
        )


class RelativeThresholdSearchArea(SearchAreaDefinitionMethod):
    """
    Defines the search area by selecting nodes whose LSM value is at or above
    a fraction of the per-scenario maximum (score >= max(axis=0) * threshold).
    threshold must be in [0, 1].
    """

    LABEL = "relative_threshold"

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
        rel_th = self._threshold if self._threshold is not None else x[0]
        return reconstructed_lsm >= reconstructed_lsm.max(axis=0) * rel_th

    @property
    def nix(self) -> int:
        return 0

    def get_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        if self._threshold is not None:
            return (np.array([]), np.array([]))
        return (np.array([self._min]), np.array([self._max]))


class AbsoluteThresholdSearchArea(SearchAreaDefinitionMethod):
    """
    Defines the search area by selecting nodes whose LSM score is at or above
    a fixed absolute threshold value (score >= threshold).
    """

    LABEL = "absolute_threshold"

    def __init__(
        self,
        threshold: Optional[float] = None,
        min_threshold: float = 0.0,
        max_threshold: float = 1.0,
    ):
        if threshold is not None:
            _check_finite(threshold, "threshold")
        else:
            _check_bounds(min_threshold, max_threshold, bounded=False)
        self._threshold = threshold
        self._min = min_threshold
        self._max = max_threshold

    def _execute(self, reconstructed_lsm: np.ndarray, x: np.ndarray) -> np.ndarray:
        thresh = self._threshold if self._threshold is not None else x[0]
        return reconstructed_lsm >= thresh

    @property
    def nix(self) -> int:
        return 0

    def get_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        if self._threshold is not None:
            return (np.array([]), np.array([]))
        return (np.array([self._min]), np.array([self._max]))
