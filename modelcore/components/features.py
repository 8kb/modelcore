"""
Features: the registered tricks a host carries in its `features` list. A new trick is a new
`#type` here (or anywhere that registers one) -- never a new block class and never a format change.
See modelcore.components.contracts.BaseFeature for the protocol. Hook points and their shapes:

  block  `residual_in`(x, x0) -> x          before the mixer: reshape the residual stream
  attention `values`(v, x, idx) -> v         after V is projected, before RoPE/norm
  attention `output`(y, x) -> y              per-head attention output (B, T, H, D), before c_proj
  stack  `after_block`(x, i, state) -> x     after block i; `state` is this forward pass's dict
  stack  `finish`(x, state) -> x             after the last block, before the final norm
"""
import torch
import torch.nn as nn

from modelcore.catalog import register_component
from modelcore.components.contracts import BaseFeature
from modelcore.components.linear import Linear


def _is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


# -- output_gate ---------------------------------------------------------------------------------

GATE_GRANULARITIES = ("head", "element", "block")


def _validate_output_gate(params, ctx):
    errors = []
    granularity = params.get("granularity")
    block_size = params.get("block_size")
    in_channels = params.get("in_channels")
    n_embd = ctx.get("n_embd")
    if granularity not in GATE_GRANULARITIES:
        errors.append(f"granularity must be one of {list(GATE_GRANULARITIES)}, got {granularity!r}")
    elif granularity == "block":
        if not _is_int(block_size) or block_size <= 0:
            errors.append(f"block_size must be a positive integer for granularity 'block', got {block_size!r}")
    elif block_size is not None:
        errors.append(f"block_size must be null for granularity {granularity!r}, got {block_size!r}")
    if in_channels is not None:
        if not _is_int(in_channels) or in_channels < 1 or (n_embd is not None and in_channels > n_embd):
            errors.append(f"in_channels must be null or an integer in [1, n_embd={n_embd}], got {in_channels!r}")
    return errors


@register_component("output_gate", needs=("n_embd",), validate=_validate_output_gate)
class OutputGate(BaseFeature):
    """Output gate of attention (Qwen "Gated Attention", G1 placement): after SDPA and before
    c_proj, y = y * sigmoid(W_g . x[..., :in_channels]) where x is the block's normed mixer
    input. granularity picks how many gates each head gets: `head` one, `element` one per output
    channel (head_dim), `block` one per contiguous block_size outputs (head_dim // block_size).
    in_channels=None means all n_embd residual channels feed the gate.

    The projection is built in bind(): a nested spec is built before its parent, so the gate cannot
    know n_head/head_dim yet. Being a Linear, the projection gets the default "matrix" role and is
    counted by stats.num_matmul_params."""
    HOOKS = ("output",)

    def __init__(self, n_embd, granularity, block_size, in_channels):
        super().__init__()
        self.granularity = granularity
        self.block_size = block_size
        self.in_channels = n_embd if in_channels is None else in_channels
        self.proj = None
        self.gates_per_head = None

    def bind(self, host):
        if self.granularity == "head":
            g = 1
        elif self.granularity == "element":
            g = host.head_dim
        else:
            assert host.head_dim % self.block_size == 0, (
                f"head_dim ({host.head_dim}) must be divisible by block_size ({self.block_size})")
            g = host.head_dim // self.block_size
        self.gates_per_head = g
        self.proj = Linear(self.in_channels, host.n_head * g, bias=False)

    @torch.no_grad()
    def init_weights(self):
        s = 3**0.5 * self.in_channels**-0.5  # same scheme as c_q
        torch.nn.init.uniform_(self.proj.weight, -s, s)

    def output(self, y, x):
        B, T, H, D = y.shape
        G = self.gates_per_head
        g = torch.sigmoid(self.proj(x[..., :self.in_channels])).view(B, T, H, G, 1)
        return (y.reshape(B, T, H, G, D // G) * g).view(B, T, H, D)


# -- value_embed ---------------------------------------------------------------------------------

def _validate_value_embed(params, ctx):
    gate_channels = params.get("gate_channels")
    n_embd = ctx.get("n_embd")
    if not _is_int(gate_channels) or gate_channels < 1 or (n_embd is not None and gate_channels > n_embd):
        return [f"gate_channels must be an integer in [1, n_embd={n_embd}], got {gate_channels!r}"]
    return []


@register_component("value_embed", needs=("padded_vocab_size", "runtime"), validate=_validate_value_embed)
class ValueEmbed(BaseFeature):
    """Value residual (ResFormer-style): a per-token embedding table added to V through an
    input-dependent gate per KV head, v = v + 3*sigmoid(gate(x[..., :gate_channels])) * embed(idx).
    A KV-sharing consumer layer has no V of its own, so the host rejects this feature there.
    The table is a value_embedding-role parameter (AdamW, its own learning rate); the gate is a
    Linear and so a "matrix" like it always was."""
    HOOKS = ("values",)
    PARAM_ROLES = {"embed": "value_embedding"}

    def __init__(self, gate_channels, padded_vocab_size, runtime):
        super().__init__()
        self.gate_channels = gate_channels
        self.padded_vocab_size = padded_vocab_size
        self.runtime = runtime
        self.embed = None
        self.gate = None

    def bind(self, host):
        assert host.produces_kv, "a KV-sharing consumer layer cannot have its own value embedding"
        self.n_embd = host.n_embd
        self.n_kv_head, self.head_dim = host.n_kv_head, host.head_dim
        self.embed = nn.Embedding(self.padded_vocab_size, host.n_kv_head * host.head_dim)
        self.gate = Linear(self.gate_channels, host.n_kv_head, bias=False)

    @torch.no_grad()
    def init_weights(self):
        s = 3**0.5 * self.n_embd**-0.5  # init like c_v: uniform with the same std
        torch.nn.init.uniform_(self.embed.weight, -s, s)
        if self.runtime.compute_dtype != torch.float16:
            self.embed.to(dtype=self.runtime.compute_dtype)
        # Gate weights init with small positive values so gates start slightly above neutral
        torch.nn.init.uniform_(self.gate.weight, 0.0, 0.02)

    def values(self, v, x, idx):
        B, T = idx.shape
        ve = self.embed(idx).to(x.dtype).view(B, T, self.n_kv_head, self.head_dim)
        gate = 3 * torch.sigmoid(self.gate(x[..., :self.gate_channels]))  # (B, T, n_kv_head), range (0, 3)
        return v + gate.unsqueeze(-1) * ve


# -- resid_lambdas -------------------------------------------------------------------------------

@register_component("resid_lambdas")
class ResidLambdas(BaseFeature):
    """Per-layer residual/x0 mixing (inspired by modded-nanogpt), applied to the block input:
    resid_lambda scales the residual stream at this layer (init ~1.0 = neutral), x0_lambda blends
    the initial embedding back in (init ~0.0 = disabled). Both inits are concrete config values,
    decided once outside modelcore."""
    HOOKS = ("residual_in",)
    PARAM_ROLES = {"resid_lambda": "resid_scalar", "x0_lambda": "x0_scalar"}

    def __init__(self, resid_lambda_init, x0_lambda_init):
        super().__init__()
        self.resid_lambda = nn.Parameter(torch.empty(()))  # fake init, real init in init_weights()
        self.x0_lambda = nn.Parameter(torch.empty(()))     # fake init, real init in init_weights()
        self._resid_lambda_init = resid_lambda_init
        self._x0_lambda_init = x0_lambda_init

    @torch.no_grad()
    def init_weights(self):
        self.resid_lambda.fill_(self._resid_lambda_init)
        self.x0_lambda.fill_(self._x0_lambda_init)

    def residual_in(self, x, x0):
        return self.resid_lambda * x + self.x0_lambda * x0


# -- backout -------------------------------------------------------------------------------------

def _validate_backout(params, ctx):
    layer = params.get("backout_layer")
    if not _is_int(layer) or layer < 0:
        return [f"backout_layer must be a non-negative integer, got {layer!r}"]
    return []


@register_component("backout", validate=_validate_backout)
class Backout(BaseFeature):
    """A mid-depth "backout": the residual stream after block `backout_layer` is remembered and
    subtracted (scaled by a learned lambda) once the last block is done, before the final norm.
    Lives on the stack, which owns the layer loop; the lambda is this feature's own parameter."""
    HOOKS = ("after_block", "finish")
    PARAM_ROLES = {"backout_lambda": "backout_scalar"}

    def __init__(self, backout_layer, backout_lambda_init):
        super().__init__()
        self.backout_layer = backout_layer
        self.backout_lambda = nn.Parameter(torch.empty(()))  # fake init, real init in init_weights()
        self._backout_lambda_init = backout_lambda_init

    @torch.no_grad()
    def init_weights(self):
        self.backout_lambda.fill_(self._backout_lambda_init)

    def after_block(self, x, i, state):
        if i == self.backout_layer:
            state["backout"] = x
        return x

    def finish(self, x, state):
        x_backout = state.get("backout")
        if x_backout is not None:
            x = x - self.backout_lambda.to(x.dtype) * x_backout
        return x
