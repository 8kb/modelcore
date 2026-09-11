"""
Delta modules: the trainable update a single modelcore.peft.apply.AdapterLinear applies on top of
its base weight. Registered by name via register_adapter; a new method is a new class here plus
one decorator, nothing else changes.

Every delta implements two methods:

    forward(self, x, y, base_weight) -> y'
        x is AdapterLinear's own input; y is the running output (the base projection, or the
        previous delta's output if more than one adapter is stacked on the same target);
        base_weight is the target's frozen-or-not base nn.Linear.weight, already cast to x's
        dtype. Most deltas (LoRA) only need x and add to y; a delta whose math is defined in
        weight-space rather than output-space (DoRA) needs base_weight too and returns a fresh
        y computed from an effective weight, discarding whatever y it was passed -- so stacking
        a weight-space delta after another delta on the same target silently drops the earlier
        one's contribution; avoid that combination (validated nowhere yet -- see
        modelcore/docs/architecture.md's "Adapters in the config tree" for the caveat).

    merge_into(self, weight) -> weight'
        Returns the base weight with this delta's contribution folded in, for
        modelcore.peft.apply.merge_adapters. Must be forward's fixed point: applying forward with
        the returned weight as base_weight and no further deltas reproduces the same y.

init_adapter_weights(self, base_weight=None) is a *separate* method from modelcore's usual
init_weights() component protocol (see modelcore.components.*), called by Model.init_weights() in
its own pass over every module that defines it (see modelcore/model.py) -- deliberately distinct
so it can't collide with a wrapped base Linear's own init_weights, and so it runs after the base
weight already has its real values (DoRA's magnitude init reads them).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from modelcore.components.linear import Linear
from modelcore.peft.registry import register_adapter


@register_adapter("lora")
class LoRADelta(nn.Module):
    """Low-rank update: y + (alpha/r) * B(A(dropout(x))). A/B are modelcore Linears (not bare
    Parameters) so they inherit the compute-dtype cast and FLOPs accounting for free; their role
    is overridden to "adapter" below (see PARAM_ROLES) rather than falling through to Linear's
    "matrix" default, since a rank-r factor is the wrong shape for Muon's Newton-Schulz step (see
    modelcore.manager.ModelManager.create_optimizer). B is zero-initialized so a freshly-applied
    adapter is an exact forward no-op until trained -- the original LoRA paper's convention, and
    what makes "hand-add a new adapter entry, reload" safe."""
    PARAM_ROLES = {"lora_A": "adapter", "lora_B": "adapter"}

    def __init__(self, in_features, out_features, r, alpha=None, dropout=0.0):
        super().__init__()
        assert r > 0, f"LoRA rank must be positive, got {r}"
        self.r = r
        self.alpha = alpha if alpha is not None else r
        self.scale = self.alpha / self.r
        self.lora_A = Linear(in_features, r, bias=False)
        self.lora_B = Linear(r, out_features, bias=False)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    @torch.no_grad()
    def init_adapter_weights(self, base_weight=None):
        # Same Uniform-for-matched-Normal-std convention as every other modelcore init (see e.g.
        # CausalSelfAttention.init_weights) -- B stays zero regardless of A's init.
        s = 3**0.5 * self.lora_A.in_features**-0.5
        nn.init.uniform_(self.lora_A.weight, -s, s)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x, y, base_weight):
        return y + self.scale * self.lora_B(self.lora_A(self.drop(x)))

    def merge_into(self, weight):
        return weight + self.scale * (self.lora_B.weight @ self.lora_A.weight)


@register_adapter("dora")
class DoRADelta(LoRADelta):
    """Weight-Decomposed Low-Rank Adaptation (https://arxiv.org/abs/2402.09353): reparameterizes
    the adapted weight as magnitude * direction, where direction = normalize_rows(W0 + BA) and
    magnitude is a learned per-output-channel scalar initialized from ||W0||_row so DoRA also
    starts as an exact no-op. This is inherently a weight-space computation (not an output-space
    addition to y) -- see the module docstring's forward() contract -- so it needs base_weight and
    ignores whatever y it was passed."""
    PARAM_ROLES = {"lora_A": "adapter", "lora_B": "adapter", "magnitude": "adapter_scalar"}

    def __init__(self, in_features, out_features, r, alpha=None, dropout=0.0):
        super().__init__(in_features, out_features, r, alpha=alpha, dropout=dropout)
        self.magnitude = nn.Parameter(torch.empty(out_features))

    @torch.no_grad()
    def init_adapter_weights(self, base_weight=None):
        super().init_adapter_weights(base_weight)
        assert base_weight is not None, "DoRADelta needs the base weight to init its magnitude"
        self.magnitude.copy_(base_weight.float().norm(dim=1).to(self.magnitude.dtype))

    def merge_into(self, weight):
        direction = weight + self.scale * (self.lora_B.weight @ self.lora_A.weight)
        dir_norm = direction.norm(dim=1, keepdim=True).clamp_min(1e-6)
        return self.magnitude.unsqueeze(1) * direction / dir_norm

    def forward(self, x, y, base_weight):
        effective_weight = self.merge_into(base_weight)
        return F.linear(x, effective_weight.to(x.dtype))
