from .core import DDLeakLocMethod
from ._impl.two_steps_dd_ll import TwoStepsDDLeakLocMehod

_REGISTRY: dict[str, type[DDLeakLocMethod]] = {
    cls.LABEL: cls
    for cls in [
        TwoStepsDDLeakLocMehod,
    ]
}

def get_dd_leak_loc_method(config: dict) -> DDLeakLocMethod:
    label = config.get("name", "")
    cls = _REGISTRY.get(label)
    if cls is None:
        raise ValueError(
            f"Unknown DD leak localization method: {label!r}. "
            f"Known labels: {list(_REGISTRY)}"
        )
    return cls.from_config(config.get("args", {}))
