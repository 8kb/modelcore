"""
ModelManager: the one entrypoint modelcore exposes. Everything a caller needs to create, load,
save, or validate a model or its optimizer, or measure a config's cost, goes through here.
Nothing else in modelcore (components, composers, catalog, roles, stats' free functions,
modelcore.precision.fp8) is meant to be used directly from outside the package -- see the module
docstrings for why each exists, but ModelManager is the seam.
"""
from contextlib import contextmanager
from dataclasses import dataclass

import torch

from modelcore.cache import KVCache
from modelcore.config.spec import ModelConfig
from modelcore.config.validate import validate_config as _validate_config
from modelcore.errors import ValidationReport
from modelcore.generate import Decoder
from modelcore.model import Model
from modelcore.optim import MuonAdamW
from modelcore.roles import build_param_groups, collect_param_roles
from modelcore.runtime import DEFAULT_RUNTIME, Runtime
from modelcore.stats import (
    ModelStats, estimate_flops, has_sliding_window as _has_sliding_window, kv_cache_spec as _kv_cache_spec,
    num_matmul_params as _num_matmul_params, shape_summary as _shape_summary,
)


@dataclass(frozen=True)
class Fp8Report:
    """What enable_fp8 did, for the caller's log line -- see ModelManager.enable_fp8."""
    num_linear: int
    num_converted: int
    num_skipped: int


@dataclass
class OptimizerHparams:
    """Numeric dials for create_optimizer(); the role -> hyperparameters *policy* itself (which
    roles exist, their relative learning rates, AdamW vs Muon, and -- load-bearing -- the order
    they're iterated in, which is the on-disk optimizer param_group layout) lives in
    ModelManager.create_optimizer, not here."""
    unembedding_lr: float = 0.004
    embedding_lr: float = 0.2
    matrix_lr: float = 0.02
    scalar_lr: float = 0.5
    weight_decay: float = 0.0


class ModelManager:
    def __init__(self, runtime: Runtime | None = None):
        self.runtime = runtime or DEFAULT_RUNTIME

    # -- config --

    def config_from_dict(self, d: dict) -> ModelConfig:
        return ModelConfig.from_dict(d)

    def config_to_dict(self, config: ModelConfig) -> dict:
        return config.to_dict()

    def validate_config(self, config: ModelConfig) -> ValidationReport:
        return _validate_config(config)

    def _require_valid(self, config: ModelConfig) -> None:
        report = self.validate_config(config)
        if not report.ok:
            raise ValueError(f"invalid model config:\n{report}")

    # -- create --

    def create_model(self, config: ModelConfig, *, device, seed: int | None = None) -> Model:
        self._require_valid(config)
        with torch.device("meta"):
            model = Model(config, runtime=self.runtime)
        model.to_empty(device=device)
        if seed is not None:
            torch.manual_seed(seed)
        model.init_weights()
        return model

    def create_optimizer(self, model: Model, hparams: OptimizerHparams | None = None) -> MuonAdamW:
        """One policy table for every model modelcore can build, regardless of which roles a
        given tree actually produces (build_param_groups skips a policy role with no params
        present) -- covers every role any currently-cataloged component can emit."""
        hparams = hparams or OptimizerHparams()
        dmodel_lr_scale = (model.config.n_embd / 768) ** -0.5
        self.runtime.log(f"Scaling the LR for the AdamW parameters ∝1/√({model.config.n_embd}/768) = {dmodel_lr_scale:.6f}")
        policy = {
            "unembedding": dict(kind='adamw', lr=hparams.unembedding_lr * dmodel_lr_scale, betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01),
            "embedding": dict(kind='adamw', lr=hparams.embedding_lr * dmodel_lr_scale, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001),
            "value_embedding": dict(kind='adamw', lr=hparams.embedding_lr * dmodel_lr_scale * 0.5, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.01),
            "resid_scalar": dict(kind='adamw', lr=hparams.scalar_lr * 0.01, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.05),
            "x0_scalar": dict(kind='adamw', lr=hparams.scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
            "smear": dict(kind='adamw', lr=0.2, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0),
            "backout_scalar": dict(kind='adamw', lr=0.2, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0),
            "matrix": dict(kind='muon', lr=hparams.matrix_lr, momentum=0.95, ns_steps=5, beta2=0.9, weight_decay=hparams.weight_decay),
        }
        param_groups = build_param_groups(collect_param_roles(model), policy)
        optimizer = MuonAdamW(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    # -- load --

    def load_model(self, store, *, device, config: ModelConfig | None = None, train: bool = False) -> Model:
        if config is None:
            config = self.config_from_dict(store.read_config())
        self._require_valid(config)
        state = store.read_model_state(map_location=device)
        with torch.device("meta"):
            model = Model(config, runtime=self.runtime)
        model.to_empty(device=device)
        # Some buffers (e.g. rotary cos/sin) are persistent=False -- never saved to a checkpoint
        # -- so they need real values from init_weights() before load_state_dict overwrites
        # everything else.
        model.init_weights()
        model.load_state_dict(state, strict=True, assign=True)
        model.train(train)
        return model

    def load_optimizer(self, model: Model, store, *, rank: int = 0,
                        hparams: OptimizerHparams | None = None) -> MuonAdamW | None:
        """Loads just the optimizer shard for a given rank; returns None if the store has none
        (not every checkpoint saves optimizer state)."""
        state = store.read_optimizer_state(rank=rank, map_location=model.get_device())
        if state is None:
            return None
        optimizer = self.create_optimizer(model, hparams)
        optimizer.load_state_dict(state)
        return optimizer

    # -- save --

    def save_model(self, model: Model, store) -> None:
        store.write_config(self.config_to_dict(model.config))
        store.write_model_state(model.state_dict())

    def save_optimizer(self, optimizer: MuonAdamW, store, *, rank: int = 0) -> None:
        store.write_optimizer_state(optimizer.state_dict(), rank=rank)

    # -- stats & runtime helpers --

    def stats(self, config: ModelConfig) -> ModelStats:
        """Computed from a meta-device model -- shapes/dtypes only, no real weights ever
        allocated, so this is cheap regardless of model size."""
        self._require_valid(config)
        with torch.device("meta"):
            model = Model(config, runtime=self.runtime)
        layer_specs = model.layer_specs()
        matmul_params = _num_matmul_params(model)
        params_by_role = {
            role: sum(p.numel() for p in params)
            for role, params in collect_param_roles(model).items()
        }
        return ModelStats(
            n_layer=config.n_layer,
            params_by_role=params_by_role,
            num_params=sum(params_by_role.values()),
            num_matmul_params=matmul_params,
            layer_specs=layer_specs,
            kv_cache_spec=_kv_cache_spec(layer_specs),
            shape_summary=_shape_summary(config, layer_specs),
            flops_per_token=estimate_flops(layer_specs, matmul_params, config.sequence_len),
            has_sliding_window=_has_sliding_window(layer_specs, config.sequence_len),
            _kv_dtype_itemsize=self.runtime.compute_dtype.itemsize,
        )

    def new_kv_cache(self, model: Model, *, batch_size: int, seq_len: int, device=None) -> KVCache:
        spec = _kv_cache_spec(model.layer_specs())
        return KVCache(
            batch_size=batch_size, seq_len=seq_len, device=device or model.get_device(),
            dtype=self.runtime.compute_dtype, **spec,
        )

    def new_decoder(self, model: Model, tokens: list, *, num_samples: int = 1,
                     max_tokens: int | None = None, device=None) -> Decoder:
        """Batch-1 prefill of tokens, replicated into an num_samples-row KV-cached decoder --
        see modelcore.generate.Decoder. The generic (tokenizer-agnostic) half of a cached
        autoregressive generation loop."""
        return Decoder(model, self, tokens, num_samples=num_samples, max_tokens=max_tokens, device=device)

    # -- precision --

    def enable_fp8(self, model: Model, *, recipe: str = "tensorwise", align: int = 16, min_dim: int = 128) -> Fp8Report:
        """Converts every eligible modelcore.components.linear.Linear in model to
        modelcore.precision.fp8.Float8Linear in place. Eligible = dims divisible by `align`
        (hardware requirement) and at least `min_dim` (below that, quantization overhead
        dominates the matmul it's supposed to speed up). Safe to call on a model whose optimizer
        hasn't been built yet -- create_optimizer/collect_param_roles see Float8Linear as an
        ordinary "matrix"-role Linear subclass, since that's what it is."""
        from modelcore.components.linear import Linear
        from modelcore.precision.fp8 import (
            Float8Linear, Float8LinearConfig, convert_to_float8_training, default_module_filter,
        )
        Float8LinearConfig.from_recipe_name(recipe)  # validates recipe; only "tensorwise" today
        num_linear = sum(1 for m in model.modules() if isinstance(m, Linear))
        module_filter_fn = lambda mod, fqn: default_module_filter(mod, fqn, align=align, min_dim=min_dim)
        convert_to_float8_training(model, module_filter_fn=module_filter_fn, runtime=self.runtime)
        num_converted = sum(1 for m in model.modules() if isinstance(m, Float8Linear))
        return Fp8Report(num_linear=num_linear, num_converted=num_converted, num_skipped=num_linear - num_converted)

    @contextmanager
    def fp8_disabled(self, model: Model):
        """Temporarily swaps every Float8Linear in model back to a plain Linear sharing the same
        weight/bias, for full-precision eval -- and restores them on exit. A no-op (still a valid
        context manager) when model has no Float8Linear at all."""
        from modelcore.components.linear import Linear
        from modelcore.precision.fp8 import find_fp8_locations
        locations = find_fp8_locations(model)
        if not locations:
            yield
            return
        for parent, attr_name, fp8_module in locations:
            # meta device: avoid a real allocation for a shell that's about to share weight/bias
            linear = Linear(
                fp8_module.in_features, fp8_module.out_features,
                bias=fp8_module.bias is not None, device="meta", dtype=fp8_module.weight.dtype,
            )
            linear.weight = fp8_module.weight
            if fp8_module.bias is not None:
                linear.bias = fp8_module.bias
            setattr(parent, attr_name, linear)
        try:
            yield
        finally:
            for parent, attr_name, fp8_module in locations:
                setattr(parent, attr_name, fp8_module)
