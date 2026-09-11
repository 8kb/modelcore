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
from modelcore.evaluate import evaluate_bpb as _evaluate_bpb
from modelcore.generate import Decoder
from modelcore.model import Model
from modelcore.optim import MuonAdamW
from modelcore.roles import build_param_groups, collect_param_roles
from modelcore.runtime import DEFAULT_RUNTIME, Runtime
from modelcore.stats import (
    ModelStats, estimate_flops, has_sliding_window as _has_sliding_window, kv_cache_spec as _kv_cache_spec,
    num_matmul_params as _num_matmul_params, shape_summary as _shape_summary,
)


def _disabled_adapter_matmul_params(model) -> int:
    """Sum of every disabled adapter delta's own Linear-weight params (lora_A/lora_B, and any
    future delta's matmul-shaped params) -- what ModelManager.stats subtracts from
    modelcore.stats.num_matmul_params so flops_per_token only charges for deltas that actually run
    forward. Module-level (not a ModelManager method) and lazily importing modelcore.peft, same
    convention as enable_fp8/fp8_disabled's local imports below."""
    from modelcore.components.linear import Linear
    from modelcore.peft import find_adapters
    total = 0
    for _, module in find_adapters(model):
        for name, delta in module.deltas.items():
            if not module.enabled.get(name, True):
                total += sum(m.weight.numel() for m in delta.modules() if isinstance(m, Linear))
    return total


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
    adapter_lr: float = 0.002        # LoRA/DoRA A/B factors -- a starting guess, not yet swept
    adapter_scalar_lr: float = 0.02  # DoRA's per-channel magnitude -- likewise unswept


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
            # Appended last, deliberately: the policy dict's iteration order is the on-disk
            # param_group layout (see modelcore.roles.build_param_groups), so a checkpoint saved
            # before these roles existed still loads its optimizer shard positionally correctly.
            # AdamW, not Muon: a rank-r LoRA/DoRA factor is the wrong shape for Muon's
            # Newton-Schulz/Polar-Express orthogonalization step.
            "adapter": dict(kind='adamw', lr=hparams.adapter_lr * dmodel_lr_scale, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0),
            "adapter_scalar": dict(kind='adamw', lr=hparams.adapter_scalar_lr * dmodel_lr_scale, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0),
        }
        param_groups = build_param_groups(collect_param_roles(model), policy)
        optimizer = MuonAdamW(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    # -- load --

    def load_model(self, store, *, device, config: ModelConfig | None = None, train: bool = False) -> Model:
        """`config`, if given, overrides what's stored (store.read_config() is then not even
        read) -- this is how a caller hand-edits a checkpoint's adapters and reloads: build the
        edited ModelConfig (e.g. via config_from_dict on a hand-modified dict) and pass it here.

        When the resulting config has adapters, the load is reconciling rather than strict: a key
        under some AdapterLinear's `deltas` (see modelcore.peft.apply.adapter_state_keys -- the
        ".deltas." substring is the load-bearing convention both sides share) that's missing from
        the checkpoint is left at its just-computed init value (a newly hand-added adapter); one
        present in the checkpoint but not in the model is dropped (a removed adapter). Any other
        missing/unexpected key is still a hard error -- only adapter keys may legitimately differ.
        A config with no adapters takes the exact original strict=True path, byte-identical to
        before this existed."""
        if config is None:
            config = self.config_from_dict(store.read_config())
        self._require_valid(config)
        state = store.read_model_state(map_location=device)
        with torch.device("meta"):
            model = Model(config, runtime=self.runtime)
        model.to_empty(device=device)
        # Some buffers (e.g. rotary cos/sin) are persistent=False -- never saved to a checkpoint
        # -- so they need real values from init_weights() before load_state_dict overwrites
        # everything else. Also the only place an adapter's own weights get their real (non-meta)
        # init values, for whichever adapters aren't found in `state` below.
        model.init_weights()

        if not config.adapters:
            model.load_state_dict(state, strict=True, assign=True)
            model.train(train)
            return model

        result = model.load_state_dict(state, strict=False, assign=True)
        is_adapter_key = lambda k: ".deltas." in k
        unexplained_missing = [k for k in result.missing_keys if not is_adapter_key(k)]
        unexplained_unexpected = [k for k in result.unexpected_keys if not is_adapter_key(k)]
        if unexplained_missing or unexplained_unexpected:
            raise RuntimeError(
                f"checkpoint state dict does not match model config "
                f"(missing={unexplained_missing}, unexpected={unexplained_unexpected})"
            )
        added = [k for k in result.missing_keys if is_adapter_key(k)]
        dropped = [k for k in result.unexpected_keys if is_adapter_key(k)]
        if added:
            self.runtime.log(f"load_model: {len(added)} adapter param(s) not in the checkpoint, kept at their init value")
        if dropped:
            self.runtime.log(f"load_model: {len(dropped)} adapter param(s) in the checkpoint dropped (not in the current config)")
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
        allocated, so this is cheap regardless of model size.

        A disabled adapter delta's params still count toward num_params/params_by_role (they
        exist on disk and take up real memory regardless of whether they currently run), but are
        subtracted out of matmul_params/flops_per_token: a delta that's off doesn't run its
        matmul, so charging FLOPs for it would overstate the actual forward cost."""
        self._require_valid(config)
        with torch.device("meta"):
            model = Model(config, runtime=self.runtime)
        layer_specs = model.layer_specs()
        matmul_params = _num_matmul_params(model) - _disabled_adapter_matmul_params(model)
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

    # -- evaluate --

    def evaluate_bpb(self, model: Model, batches, steps: int, token_bytes, *, bos_token_id=None,
                      doc_masking_max_docs_per_row: int | None = None, padding_id: int | None = None) -> float:
        """Bits-per-byte over `steps` batches -- see modelcore.evaluate.evaluate_bpb for the full
        contract (in particular what token_bytes must be and why doc masking is optional)."""
        return _evaluate_bpb(
            model, batches, steps, token_bytes, bos_token_id=bos_token_id,
            doc_masking_max_docs_per_row=doc_masking_max_docs_per_row, padding_id=padding_id,
        )

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

    # -- adapters --

    def merge_adapters(self, model: Model) -> int:
        """Folds every currently-enabled adapter delta into its target's base weight and swaps
        that target back to a plain modelcore.components.linear.Linear -- see
        modelcore.peft.apply.merge_adapters for what "fold" means for a given delta type. Returns
        the number of targets merged. Intended for serving/eval once the adapter list is done
        being edited: model.config still lists the (now-merged) adapters unchanged, so building a
        fresh unmerged model from the same config afterward is still possible -- this only changes
        the in-memory module tree, not model.config, and merge_adapters itself never saves
        anything."""
        from modelcore.peft import merge_adapters as _merge_adapters
        return _merge_adapters(model)

    @contextmanager
    def adapters_disabled(self, model: Model):
        """Temporarily disables every adapter delta in model (restoring the exact prior
        enabled/disabled state on exit) -- for an eval that needs the base model's own behavior
        underneath its adapters. A no-op (still a valid context manager) when model has no
        adapters at all. Mirrors fp8_disabled's shape, but needs no module swap: `enabled` is a
        plain dict (see modelcore.peft.apply.AdapterLinear), so toggling it is enough -- disabling
        also sets requires_grad_(False) on the affected deltas (see AdapterLinear.set_enabled),
        which this restores too."""
        from modelcore.peft import find_adapters
        locations = find_adapters(model)
        if not locations:
            yield
            return
        saved = [dict(module.enabled) for _, module in locations]
        for _, module in locations:
            for name in list(module.enabled):
                module.set_enabled(name, False)
        try:
            yield
        finally:
            for (_, module), state in zip(locations, saved):
                for name, was_enabled in state.items():
                    module.set_enabled(name, was_enabled)
