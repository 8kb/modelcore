"""
modelcore -- a standalone model subsystem: configs, architectures-as-data, and the machinery to
create/load/save models and optimizers and compute their stats. Knows nothing about a host
application's checkpoint naming, tokenizers, or CLI flags; see modelcore/docs/architecture.md for
the full contract. A host adapts its own preset/depth-dial layer and checkpoint naming onto it.

ModelManager is the one entrypoint; ModelConfig/ComponentSpec/AdapterSpec, ModelStats,
ValidationReport, OptimizerHparams, ArtifactStore/FileSystemStore, and KVCache are the value types
that cross its boundary; Decoder/generate_with_tools, the runtime helpers, derive_training_plan,
the step schedules, build_doc_args and find_adapters are the other names re-exported below
(`__all__` is the public surface). Everything else (components, composers, catalog, roles) is
internal; modelcore.config.upgrade and modelcore.convert are public as modules.

Importing this package (or modelcore.manager) triggers every built-in component/composer's
@register_component decorator, via modelcore.components/modelcore.composers -- see catalog.py.
"""
import modelcore.components  # noqa: F401 -- import for @register_component side effects
import modelcore.composers  # noqa: F401 -- import for @register_component side effects

from modelcore.cache import KVCache
from modelcore.config.spec import (
    AdapterSpec, AttentionLayerSpec, ComponentSpec, ModelConfig, RecurrentLayerSpec,
)
from modelcore.errors import ConfigError, ValidationReport
from modelcore.generate import (
    Decoder, ToolSpec, collect_batch, collect_batch_multi, generate_with_tools, sample_next_token,
)
from modelcore.kernels.flash_attn import build_doc_args
from modelcore.manager import Fp8Report, ModelManager, OptimizerHparams
from modelcore.model import Model
from modelcore.optim.schedules import lr_multiplier, muon_momentum
from modelcore.peft import find_adapters
from modelcore.runtime import (
    DEFAULT_RUNTIME, Runtime, autodetect_device_type, compute_cleanup, compute_init,
    peak_bandwidth, peak_flops,
)
from modelcore.scaling import TrainingPlan, derive_training_plan
from modelcore.stats import ModelStats
from modelcore.store import ArtifactStore, FileSystemStore, last_step

__all__ = [
    "ModelManager", "OptimizerHparams", "Fp8Report",
    "ModelConfig", "ComponentSpec", "AttentionLayerSpec", "RecurrentLayerSpec", "AdapterSpec",
    "Model", "ModelStats", "KVCache",
    "Decoder", "sample_next_token",
    "ToolSpec", "generate_with_tools", "collect_batch", "collect_batch_multi",
    "ConfigError", "ValidationReport",
    "ArtifactStore", "FileSystemStore", "last_step",
    "Runtime", "DEFAULT_RUNTIME", "compute_init", "compute_cleanup", "autodetect_device_type",
    "peak_flops", "peak_bandwidth",
    "TrainingPlan", "derive_training_plan",
    "build_doc_args", "lr_multiplier", "muon_momentum", "find_adapters",
]
