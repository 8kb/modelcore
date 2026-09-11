"""
Adapter-type registry: maps a name ("lora", "dora", ...) to the delta class that implements it --
the modelcore.peft analog of modelcore.catalog's component registry. Deliberately a separate,
smaller registry rather than reusing modelcore.catalog: a delta isn't a modelcore.catalog
component (it never goes through build_component's `needs` context-injection), and its
constructor kwargs are just modelcore.config.spec.AdapterSpec.params plus the
in_features/out_features modelcore.peft.apply.AdapterLinear already knows from the target it
wraps.

A new PEFT method (IA3, prefix-tuning, ...) is a new delta class plus one
@register_adapter(name) decorator here -- nothing else in modelcore.peft or modelcore.manager
needs to change for it to become selectable from an AdapterSpec.type string.
"""

_ADAPTER_REGISTRY = {}  # type name -> delta class


def register_adapter(name):
    def decorator(cls):
        assert name not in _ADAPTER_REGISTRY, f"adapter type {name!r} already registered"
        _ADAPTER_REGISTRY[name] = cls
        return cls
    return decorator


def registered_adapter_types():
    return sorted(_ADAPTER_REGISTRY)


def get_adapter_delta_cls(name):
    if name not in _ADAPTER_REGISTRY:
        raise ValueError(f"Unknown adapter type {name!r}. Registered: {registered_adapter_types()}")
    return _ADAPTER_REGISTRY[name]
