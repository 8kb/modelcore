"""
Turns a ModelConfig's `adapters` (list of modelcore.config.spec.AdapterSpec) into real
AdapterLinear modules wrapping their targeted Linears, and back. Mirrors modelcore.precision.fp8's
shape: a module-tree walk that swaps a plain Linear in place via setattr(parent, attr, ...),
sharing (not copying) its weight -- see convert_to_float8_training/find_fp8_locations for the
precedent this follows.

apply_adapters is idempotent and incremental: calling it again with a spec list that's a superset
of what's already applied only adds the *new* (target, name) pairs -- an already-applied one is
left alone (its `enabled` is refreshed from the spec, everything else untouched). This is what
lets Model.__init__ splice a hand-edited adapter list onto a config without any special-casing at
load time; see modelcore.manager.ModelManager.load_model.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from modelcore.components.linear import Linear
from modelcore.peft.registry import get_adapter_delta_cls


class AdapterLinear(Linear):
    """A modelcore Linear that owns zero or more named deltas (modelcore.peft.deltas.LoRADelta,
    DoRADelta, ...) applied on top of its own base projection. Subclasses Linear (not nn.Linear)
    for the same reason modelcore.precision.fp8.Float8Linear does: Linear is the structural marker
    modelcore.roles.collect_param_roles and modelcore.stats.num_matmul_params key off.

    `self.enabled` is a plain Python dict (name -> bool), never a buffer or Parameter -- it must
    never enter state_dict(), since toggling it is the entire point of "hand-edit the config to
    disable an adapter, no state-dict surgery": two AdapterLinears loaded from the same state dict
    with different `enabled` values are otherwise byte-identical."""

    def __init__(self, in_features, out_features, bias=False):
        super().__init__(in_features, out_features, bias=bias)
        self.deltas = nn.ModuleDict()
        self.enabled = {}

    @classmethod
    def from_linear(cls, mod):
        """Meta-device shell sharing mod's weight/bias -- see Float8Linear.from_float, the
        precedent for this exact trick (avoid a real allocation for a shell about to share
        tensors with the module it's replacing)."""
        with torch.device("meta"):
            new_mod = cls(mod.in_features, mod.out_features, bias=mod.bias is not None)
        new_mod.weight = mod.weight
        if mod.bias is not None:
            new_mod.bias = mod.bias
        return new_mod

    def add_delta(self, name, delta_type, params, enabled=True):
        assert name not in self.deltas, f"adapter {name!r} already applied to this target"
        delta_cls = get_adapter_delta_cls(delta_type)
        delta = delta_cls(self.in_features, self.out_features, **params)
        delta.requires_grad_(enabled)  # see set_enabled below for why this must stay in sync
        self.deltas[name] = delta
        self.enabled[name] = enabled

    def set_enabled(self, name, enabled):
        """Also toggles requires_grad_ on the whole delta, not just the forward-time `enabled`
        flag: a disabled delta is skipped in forward (see AdapterLinear.forward), so it would
        never receive a gradient anyway -- leaving requires_grad=True would put a
        permanently-None-grad parameter into an optimizer group, which
        modelcore.optim.MuonAdamW.step() dereferences unconditionally and crashes on (see
        modelcore.roles.build_param_groups' requires_grad filter). Keeping the two in sync is
        also what makes `enabled=False` the *only* mechanism needed for "this adapter shouldn't
        train" -- there is no separate frozen-adapter path in modelcore.config.spec.ModelConfig.frozen."""
        assert name in self.deltas, f"no adapter named {name!r} on this target"
        self.deltas[name].requires_grad_(enabled)
        self.enabled[name] = enabled

    @torch.no_grad()
    def init_adapter_weights(self):
        base_weight = self.weight
        for delta in self.deltas.values():
            delta.init_adapter_weights(base_weight=base_weight)

    def forward(self, x):
        base_weight = self.weight.to(dtype=x.dtype)
        y = F.linear(x, base_weight)
        for name, delta in self.deltas.items():
            if self.enabled.get(name, True):
                y = delta(x, y, base_weight)
        return y


def _resolve_target(model, target):
    if "." in target:
        parent_name, attr = target.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
    else:
        parent, attr = model, target
    child = getattr(parent, attr)
    return parent, attr, child


def apply_adapters(model, specs):
    """Group `specs` by target FQN and, for each, ensure the targeted submodule is an
    AdapterLinear (converting a plain Linear in place if it isn't one yet) carrying every spec's
    delta. Order within a target follows `specs`' own order -- deltas apply in that order at
    forward time (see AdapterLinear.forward)."""
    for spec in specs:
        parent, attr, child = _resolve_target(model, spec.target)
        if not isinstance(child, AdapterLinear):
            assert isinstance(child, Linear), (
                f"adapter target {spec.target!r} is a {type(child).__name__}, not a modelcore Linear"
            )
            child = AdapterLinear.from_linear(child)
            setattr(parent, attr, child)
        if spec.name not in child.deltas:
            child.add_delta(spec.name, spec.type, spec.params, enabled=spec.enabled)
        else:
            child.set_enabled(spec.name, spec.enabled)


def find_adapters(model):
    """Every (fqn, AdapterLinear) in model, in module-tree order -- the read side of
    apply_adapters, used by ModelManager.stats/load_model/merge_adapters and by nanochat's
    model_info.py inventory listing."""
    return [(name, module) for name, module in model.named_modules() if isinstance(module, AdapterLinear)]


def adapter_state_keys(model):
    """Every state_dict key that exists only because of an adapter (i.e. lives under some
    AdapterLinear's `deltas`), keyed by the fixed, hardcoded ".deltas." attribute name --
    ModelManager.load_model uses this same substring convention directly rather than rebuilding a
    second model to diff against, so this helper and that convention must not drift apart."""
    keys = set()
    for fqn, module in find_adapters(model):
        for name, _ in module.deltas.named_parameters():
            keys.add(f"{fqn}.deltas.{name}")
    return keys


def merge_adapters(model):
    """Fold every *enabled* delta's contribution into its target's base weight (via that delta's
    own merge_into) and swap the target back to a plain Linear -- for serving/eval throughput once
    the adapter list is done being edited. Disabled deltas are dropped, not merged in: this
    produces the model that's currently *running*, not everything ever attached. Stacking more
    than one delta on the same target folds them in `specs` order; see deltas.py's module
    docstring for the caveat about mixing an output-space delta (LoRA) with a weight-space one
    (DoRA) on the same target.

    Returns the number of targets merged. Does not touch model.config -- the (now-merged) adapters
    are still listed there, so a fresh unmerged model can still be built from the same config."""
    locations = find_adapters(model)
    for fqn, module in locations:
        parent, attr, _ = _resolve_target(model, fqn)
        with torch.no_grad():
            weight = module.weight.clone()
            for name, delta in module.deltas.items():
                if module.enabled.get(name, True):
                    weight = delta.merge_into(weight)
            new_linear = Linear(module.in_features, module.out_features, bias=module.bias is not None)
            new_linear.to(device=weight.device, dtype=weight.dtype)
            new_linear.weight.copy_(weight)
            if module.bias is not None:
                new_linear.bias.copy_(module.bias)
        setattr(parent, attr, new_linear)
    return len(locations)


def strip_adapters(model):
    """Remove every AdapterLinear entirely, restoring the plain base Linear with none of its
    deltas' contributions applied -- "give me the true base model back", discarding adapter
    weights rather than folding them in (the inverse of merge_adapters, not its complement)."""
    locations = find_adapters(model)
    for fqn, module in locations:
        parent, attr, _ = _resolve_target(model, fqn)
        new_linear = Linear(module.in_features, module.out_features, bias=module.bias is not None)
        new_linear.to(device=module.weight.device, dtype=module.weight.dtype)
        with torch.no_grad():
            new_linear.weight.copy_(module.weight)
            if module.bias is not None:
                new_linear.bias.copy_(module.bias)
        setattr(parent, attr, new_linear)
    return len(locations)
