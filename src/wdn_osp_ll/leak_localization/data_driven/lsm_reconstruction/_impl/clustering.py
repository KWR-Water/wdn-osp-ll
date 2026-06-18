import numpy as np

from ..base import LSMReconstructionMethod


class ClusteringLSMReconstruction(LSMReconstructionMethod):
    """
    LSM reconstruction via node clustering.
    Groups network nodes into clusters and propagates observed LSM values.
    """

    LABEL = "clustering"

    def _execute(self, observed_lsm: np.ndarray, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @classmethod
    def from_config(cls, config: dict) -> "ClusteringLSMReconstruction":
        return cls()
