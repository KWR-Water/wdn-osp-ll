import numpy as np

from ..base import SearchAreaDefinitionMethod


class RelativeThresholdSearchArea(SearchAreaDefinitionMethod):
    """
    Defines the search area by selecting nodes whose LSM score is within a
    relative tolerance of the maximum score (e.g. score >= max * (1 - tol)).
    """

    LABEL = "relative_threshold"

    def _execute(self, reconstructed_lsm: np.ndarray, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @classmethod
    def from_config(cls, config: dict) -> "RelativeThresholdSearchArea":
        return cls()
