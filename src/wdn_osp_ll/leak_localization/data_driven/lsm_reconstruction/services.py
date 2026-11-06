from __future__ import annotations

from typing import TYPE_CHECKING

from .base import LSMReconstructionMethod
from ._impl.boatwright import BoatwrightLSMReconstruction
from ._impl.alves import AlvesLSMReconstruction
from ._impl.clustering import ClusteringLSMReconstruction

if TYPE_CHECKING:
    from wdn_osp_ll.hyd_analysis.data_structures import WDNBaseAnalysis

_REGISTRY: dict[str, type[LSMReconstructionMethod]] = {
    cls.LABEL: cls
    for cls in [
        BoatwrightLSMReconstruction,
        AlvesLSMReconstruction,
        ClusteringLSMReconstruction,
    ]
}


def get_lsm_reconstruction_method(
    config: dict, wdn_base_analysis: "WDNBaseAnalysis"
) -> LSMReconstructionMethod:
    label = config.get("name", "")
    cls = _REGISTRY.get(label)
    if cls is None:
        raise ValueError(
            f"Unknown LSM reconstruction method: {label!r}. "
            f"Known labels: {list(_REGISTRY)}"
        )
    return cls.from_config(config.get("args", {}), wdn_base_analysis)
