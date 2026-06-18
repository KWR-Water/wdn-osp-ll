import numpy as np

from ..base import SearchAreaDefinitionMethod


class AbsoluteThresholdSearchArea(SearchAreaDefinitionMethod):
    """
    Defines the search area by selecting nodes whose LSM score exceeds a
    fixed absolute threshold value.
    """

    LABEL = "absolute_threshold"

    def _execute(self, reconstructed_lsm: np.ndarray, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @classmethod
    def from_config(cls, config: dict) -> "AbsoluteThresholdSearchArea":
        return cls()
