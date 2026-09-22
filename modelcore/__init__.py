"""
modelcore -- a standalone model subsystem: configs, architectures-as-data, and the machinery to
create/load/save models and optimizers and compute their stats. Knows nothing about a host
application's checkpoint naming, tokenizers, or CLI flags; see modelcore/docs/architecture.md for
the full contract, and (in this repo) nanochat/architectures/ and nanochat/checkpoint_manager.py
for the layer that adapts a specific application onto it.

ModelManager is the one entrypoint; ModelConfig/ComponentSpec/AdapterSpec, ModelStats,
ValidationReport, OptimizerHparams, ArtifactStore/FileSystemStore, and KVCache are the value types
that cross its boundary. Everything else (components, composers, catalog, roles, modelcore.peft)
is internal.

Importing this package (or modelcore.manager) triggers every built-in component/composer's
@register_component decorator, via modelcore.components/modelcore.composers -- see catalog.py.
"""
import modelcore.components  # noqa: F401 -- import for @register_component side effects
import modelcore.composers  # noqa: F401 -- import for @register_component side effects

from modelcore.cache import KVCache
from modelcore.config.spec import (
    AdapterSpec, AttentionLayerSpec, ComponentSpec, ModelConfig, resolve_reference_config,
)
from modelcore.errors import ConfigError, ValidationReport
from modelcore.generate import (
    Decoder, ToolSpec, collect_batch, collect_batch_multi, generate_naive, generate_with_tools,
    sample_next_token,
)
from modelcore.manager import Fp8Report, ModelManager, OptimizerHparams
from modelcore.model import Model
from modelcore.runtime import (
    DEFAULT_RUNTIME, Runtime, autodetect_device_type, compute_cleanup, compute_init,
    peak_bandwidth, peak_flops,
)
from modelcore.scaling import TrainingPlan, derive_training_plan
from modelcore.stats import ModelStats
from modelcore.store import ArtifactStore, FileSystemStore, last_step

__all__ = [
    "ModelManager", "OptimizerHparams", "Fp8Report",
    "ModelConfig", "ComponentSpec", "AttentionLayerSpec", "AdapterSpec", "resolve_reference_config",
    "Model", "ModelStats", "KVCache",
    "Decoder", "generate_naive", "sample_next_token",
    "ToolSpec", "generate_with_tools", "collect_batch", "collect_batch_multi",
    "ConfigError", "ValidationReport",
    "ArtifactStore", "FileSystemStore", "last_step",
    "Runtime", "DEFAULT_RUNTIME", "compute_init", "compute_cleanup", "autodetect_device_type",
    "peak_flops", "peak_bandwidth",
    "TrainingPlan", "derive_training_plan",
]
