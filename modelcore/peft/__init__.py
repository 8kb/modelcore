"""
modelcore.peft -- low-rank adapters (LoRA, DoRA today; IA3/prefix-tuning fit the same seam later)
as first-class entries in the same materialized config tree modelcore.config.spec.ModelConfig
already uses for architecture. See modelcore/docs/architecture.md's "Adapters in the config tree"
for the design, and this package's own apply.py/deltas.py/registry.py docstrings for the pieces.

Not part of modelcore's public entrypoint contract on its own: Model/ModelManager call into here
(Model.__init__ applies config.adapters automatically; ModelManager.load_model, .stats, and the
new .merge_adapters/.adapters_disabled use find_adapters/adapter_state_keys/merge_adapters/
strip_adapters) so a caller normally never imports modelcore.peft directly -- it's exposed here,
mirroring modelcore.precision.fp8's Float8Linear, for a caller that wants AdapterLinear/
find_adapters/merge_adapters without going through a full save/load round trip, or that's adding
a new adapter type via register_adapter.

Importing this package registers every built-in delta type ("lora", "dora") via their
@register_adapter decorators in deltas.py -- unlike modelcore/__init__.py's eager
`import modelcore.components`, nothing imports modelcore.peft at modelcore's own package-import
time; every call site (Model.__init__, ModelManager) imports it lazily, matching
ModelManager.enable_fp8's lazy-import convention for modelcore.precision.fp8.
"""
from modelcore.peft.apply import (
    AdapterLinear, adapter_state_keys, apply_adapters, find_adapters, merge_adapters, strip_adapters,
)
from modelcore.peft.deltas import DoRADelta, LoRADelta
from modelcore.peft.registry import get_adapter_delta_cls, register_adapter, registered_adapter_types

__all__ = [
    "AdapterLinear", "apply_adapters", "find_adapters", "adapter_state_keys",
    "merge_adapters", "strip_adapters",
    "LoRADelta", "DoRADelta", "register_adapter", "get_adapter_delta_cls", "registered_adapter_types",
]
