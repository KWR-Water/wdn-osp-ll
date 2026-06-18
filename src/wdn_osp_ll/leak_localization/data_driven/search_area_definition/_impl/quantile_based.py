import numpy as np

from ..base import SearchAreaDefinitionMethod


class QuantileBasedSearchArea(SearchAreaDefinitionMethod):
    """
    Defines the search area by selecting nodes whose LSM score exceeds a
    quantile threshold of the reconstructed LSM distribution.
    """

    LABEL = "quantile"

    def _execute(self, reconstructed_lsm: np.ndarray, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @classmethod
    def from_config(cls, config: dict) -> "QuantileBasedSearchArea":
        return cls()
