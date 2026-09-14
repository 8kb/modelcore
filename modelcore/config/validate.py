"""
validate_config(): walks a ModelConfig tree and returns every problem found (modelcore.errors.
ValidationReport), never just the first. Split the way every other part of modelcore is:

- Structural checks (this module): input/body/output present, every #type registered, every
  needed build-context value available, no unknown or missing constructor params. These apply to
  any component uniformly, so core owns them.
- Semantic checks (each component's own `validate(params, ctx)`, registered alongside it in
  modelcore.catalog): "n_embd must be divisible by n_head", "a KV-sharing consumer can't have its
  own value embedding" -- things only the component itself knows the rule for.
- Cross-layer checks (this module, but expressed generically over any composer's block list, not
  hardcoded to one composer type): KV slots form a contiguous 0..M-1 range, a consumer's kv_slot
  points at an earlier producer, n_kv_head/head_dim are uniform across every attention-shaped
  block -- exactly what modelcore.stats.kv_cache_spec() requires at model-build time, checked here
  before a model is ever built.
"""
import dataclasses
import inspect

from modelcore.catalog import get_component, registered_types
from modelcore.config.spec import ComponentSpec
from modelcore.errors import ConfigError, ValidationReport


def _validate_spec(spec, ctx, path, errors):
    if not isinstance(spec, ComponentSpec):
        errors.append(ConfigError(path, f"expected a component spec, got {type(spec).__name__}"))
        return
    if spec.type not in registered_types():
        errors.append(ConfigError(f"{path}.#type", f"unknown component type {spec.type!r}; registered: {registered_types()}"))
        return
    cls, needs, validate = get_component(spec.type)

    for name in needs:
        if name not in ctx:
            errors.append(ConfigError(path, f"component {spec.type!r} needs {name!r}, not available here"))

    # Recurse into any nested ComponentSpec (or list of them) before checking this spec's own
    # constructor params, so a bad nested spec doesn't also register as a spurious "unknown
    # param" here (its param name is legitimate; its value is what's wrong).
    nested_keys = set()
    for key, value in spec.params.items():
        if isinstance(value, ComponentSpec):
            nested_keys.add(key)
            _validate_spec(value, ctx, f"{path}.{key}", errors)
        elif isinstance(value, list) and value and all(isinstance(v, ComponentSpec) for v in value):
            nested_keys.add(key)
            for i, v in enumerate(value):
                _validate_spec(v, ctx, f"{path}.{key}[{i}]", errors)

    sig = inspect.signature(cls.__init__)
    accepts_var_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    accepted = {p for p in sig.parameters if p != "self"}
    if not accepts_var_kwargs:
        unknown = set(spec.params) - accepted - set(needs)
        for key in sorted(unknown):
            errors.append(ConfigError(f"{path}.{key}", f"{spec.type!r} has no such param"))
        missing = [
            name for name, param in sig.parameters.items()
            if name != "self" and name not in needs and name not in spec.params
            and param.default is inspect.Parameter.empty
        ]
        for name in missing:
            errors.append(ConfigError(path, f"{spec.type!r} missing required param {name!r}"))

    if validate is not None:
        for message in (validate(spec.params, ctx) or []):
            errors.append(ConfigError(path, message))


def _collect_block_specs(spec, path):
    """Every ComponentSpec reachable through a "blocks"-named list, in forward-pass order, with
    its path -- however many composers arrange them (recurses into nested composers). Mirrors
    modelcore.config.spec._count_blocks's traversal convention."""
    if not isinstance(spec, ComponentSpec):
        return []
    found = []
    for key, value in spec.params.items():
        if key == "blocks" and isinstance(value, list):
            for i, v in enumerate(value):
                found.append((f"{path}.{key}[{i}]", v))
        elif isinstance(value, ComponentSpec):
            found.extend(_collect_block_specs(value, f"{path}.{key}"))
        elif isinstance(value, list):
            for i, v in enumerate(value):
                found.extend(_collect_block_specs(v, f"{path}.{key}[{i}]"))
    return found


def _validate_kv_layout(body, errors):
    """Structural cross-layer checks -- computed directly from block spec params (n_head,
    n_kv_head, window, kv_slot, produces_kv), without building a real model, so a bad config is
    caught before any tensor is allocated. Mirrors what modelcore.stats.kv_cache_spec() would
    otherwise raise an AssertionError for at model-build time."""
    blocks = _collect_block_specs(body, "body")
    if not blocks:
        return
    n_heads, n_kv_heads, slots = set(), set(), []
    for i, (path, block_spec) in enumerate(blocks):
        p = block_spec.params
        n_head = p.get("n_head")
        n_kv_head = p.get("n_kv_head", n_head)
        if n_head is not None:
            n_heads.add(n_head)
        if n_kv_head is not None:
            n_kv_heads.add(n_kv_head)
        kv_slot = p.get("kv_slot", i)
        slots.append(kv_slot if kv_slot is not None else i)
        produces_kv = p.get("produces_kv", True)
        if not produces_kv and kv_slot is not None and kv_slot >= i:
            errors.append(ConfigError(path, f"kv_slot {kv_slot} does not point at an earlier producer layer"))
    if len(n_heads) > 1:
        # head_dim = n_embd // n_head, and n_embd is one global value -- non-uniform n_head is
        # exactly non-uniform head_dim, which kv_cache_spec() also requires uniform.
        errors.append(ConfigError("body", f"non-uniform n_head across blocks: {sorted(n_heads)} (KV cache requires uniform head_dim, i.e. uniform n_head)"))
    if len(n_kv_heads) > 1:
        errors.append(ConfigError("body", f"non-uniform n_kv_head across blocks: {sorted(n_kv_heads)} (KV cache requires uniform n_kv_head)"))
    distinct_slots = sorted(set(slots))
    if distinct_slots != list(range(len(distinct_slots))):
        errors.append(ConfigError("body", f"kv slots are not a contiguous 0..M-1 range: {distinct_slots}"))


def _validate_adapters(config, errors):
    """Checks config.adapters/config.frozen: every `target`/frozen FQN resolves against the
    *built* tree, every adapter `type` is registered, and its `params` match that delta class's
    own constructor signature (reusing _validate_spec's inspect.signature convention above).

    Unlike every other check in this module, this one needs a real (meta-device -- still no
    tensor ever allocated) Model: an adapter target is a module attribute path inside a
    component's own Python class (e.g. CausalSelfAttention.c_q), not something expressible from
    ComponentSpec params alone, so there's no way to check it without building the tree it refers
    into. Skipped entirely if the base tree already has errors (building it would likely crash
    outright, e.g. a hard `assert n_embd % n_head == 0` inside a component's own __init__ that
    _validate_attention_shape above exists specifically to catch before that point)."""
    if not config.adapters and not config.frozen:
        return
    if errors:
        return
    import torch

    from modelcore.model import Model
    from modelcore.components.linear import Linear
    from modelcore.peft.registry import get_adapter_delta_cls, registered_adapter_types

    bare = dataclasses.replace(config, adapters=[], frozen=[])
    try:
        with torch.device("meta"):
            model = Model(bare)
    except Exception as e:
        errors.append(ConfigError("adapters", f"could not build the model tree to validate adapter/frozen targets: {e}"))
        return

    seen = set()
    for i, spec in enumerate(config.adapters):
        path = f"adapters[{i}]"
        try:
            target_module = model.get_submodule(spec.target)
        except AttributeError:
            errors.append(ConfigError(f"{path}.target", f"no such module {spec.target!r}"))
            continue
        if not isinstance(target_module, Linear):
            errors.append(ConfigError(f"{path}.target", f"{spec.target!r} is a {type(target_module).__name__}, not a Linear"))

        if spec.type not in registered_adapter_types():
            errors.append(ConfigError(f"{path}.type", f"unknown adapter type {spec.type!r}; registered: {registered_adapter_types()}"))
        else:
            delta_cls = get_adapter_delta_cls(spec.type)
            sig = inspect.signature(delta_cls.__init__)
            fixed = {"self", "in_features", "out_features"}
            accepts_var_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
            accepted = {p for p in sig.parameters if p not in fixed}
            if not accepts_var_kwargs:
                for key in sorted(set(spec.params) - accepted):
                    errors.append(ConfigError(f"{path}.params.{key}", f"{spec.type!r} has no such param"))
                missing = [
                    name for name, param in sig.parameters.items()
                    if name not in fixed and name not in spec.params and param.default is inspect.Parameter.empty
                ]
                for name in missing:
                    errors.append(ConfigError(f"{path}.params", f"{spec.type!r} missing required param {name!r}"))

        key = (spec.target, spec.name)
        if key in seen:
            errors.append(ConfigError(f"{path}.name", f"adapter name {spec.name!r} already used on target {spec.target!r}"))
        seen.add(key)

    for i, fqn in enumerate(config.frozen):
        try:
            model.get_submodule(fqn)
        except AttributeError:
            errors.append(ConfigError(f"frozen[{i}]", f"no such module {fqn!r}"))


def validate_config(config) -> ValidationReport:
    errors = []

    for field_name in ("n_embd", "vocab_size", "sequence_len", "pad_vocab_size_to"):
        value = getattr(config, field_name)
        if not isinstance(value, int) or value <= 0:
            errors.append(ConfigError(field_name, f"must be a positive integer, got {value!r}"))

    ctx = {
        "n_embd": config.n_embd, "vocab_size": config.vocab_size,
        "padded_vocab_size": config.padded_vocab_size, "sequence_len": config.sequence_len,
        "runtime": True,
    }
    for name, spec in config.shared.items():
        _validate_spec(spec, ctx, f"shared.{name}", errors)
        ctx[name] = True  # available for a later needs=(..., name) lookup, e.g. "rope"

    if config.input is None:
        errors.append(ConfigError("input", "must be set"))
    else:
        _validate_spec(config.input, ctx, "input", errors)

    if config.body is None:
        errors.append(ConfigError("body", "must be set"))
    else:
        _validate_spec(config.body, ctx, "body", errors)
        _validate_kv_layout(config.body, errors)

    if config.output is None:
        errors.append(ConfigError("output", "must be set"))
    else:
        _validate_spec(config.output, ctx, "output", errors)

    _validate_adapters(config, errors)

    return ValidationReport(errors)
