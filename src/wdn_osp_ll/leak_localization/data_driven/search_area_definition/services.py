from .base import SearchAreaDefinitionMethod
from ._impl.quantile_based import QuantileBasedSearchArea
from ._impl.threshold_based import AbsoluteThresholdSearchArea, RelativeThresholdSearchArea

_REGISTRY: dict[str, type[SearchAreaDefinitionMethod]] = {
    cls.LABEL: cls
    for cls in [
        QuantileBasedSearchArea,
        RelativeThresholdSearchArea,
        AbsoluteThresholdSearchArea,
    ]
}


def get_search_area_definition_method(config: dict) -> SearchAreaDefinitionMethod:
    label = config.get("name", "")
    cls = _REGISTRY.get(label)
    if cls is None:
        raise ValueError(
            f"Unknown search area definition method: {label!r}. "
            f"Known labels: {list(_REGISTRY)}"
        )
    return cls.from_config(config.get("args", {}))
