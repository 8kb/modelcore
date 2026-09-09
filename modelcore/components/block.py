import torch
import torch.nn as nn

from modelcore.catalog import register_component
from modelcore.components.attention import CausalSelfAttention
from modelcore.components.contracts import BaseBlock
from modelcore.components.mlp import MLP, SwiGLUMLP
from modelcore.components.norm import norm


def _validate_attention_shape(params, ctx):
    """Shared semantic checks for any attention-shaped block: n_embd/n_head/n_kv_head must be
    mutually consistent (the same constraints modelcore.components.attention.CausalSelfAttention
    asserts at construction time -- reported here as validation errors instead of crashing the
    build), and window must be a real window value."""
    errors = []
    n_head = params.get("n_head")
    n_kv_head = params.get("n_kv_head", n_head)
    n_embd = ctx.get("n_embd")
    if n_head and n_embd is not None and n_embd % n_head != 0:
        errors.append(f"n_embd ({n_embd}) must be divisible by n_head ({n_head})")
    if n_head and n_kv_head:
        if n_kv_head > n_head:
            errors.append(f"n_kv_head ({n_kv_head}) cannot exceed n_head ({n_head})")
        elif n_head % n_kv_head != 0:
            errors.append(f"n_head ({n_head}) must be divisible by n_kv_head ({n_kv_head})")
    window = params.get("window", -1)
    if not isinstance(window, int) or window < -1:
        errors.append(f"window must be -1 (full context) or a non-negative integer, got {window!r}")
    return errors


def _validate_gpt_block(params, ctx):
    errors = _validate_attention_shape(params, ctx)
    if params.get("has_value_embed") and not params.get("produces_kv", True):
        errors.append("a KV-sharing consumer layer (produces_kv=False) cannot have has_value_embed=True")
    return errors


def _validate_plain_block(params, ctx):
    errors = _validate_attention_shape(params, ctx)
    if not params.get("produces_kv", True) and params.get("kv_slot") is None:
        errors.append("produces_kv=False requires an explicit kv_slot pointing at the producer layer")
    return errors


@register_component("gpt_block", needs=("n_embd", "padded_vocab_size", "rope", "runtime"), validate=_validate_gpt_block)
class Block(BaseBlock):
    """CausalSelfAttention + MLP, plus the per-layer resid/x0-lambda residual mixing (inspired by
    modded-nanogpt): resid_lambda scales the residual stream at this layer (init ~1.0 = neutral),
    x0_lambda blends the initial embedding back in (init ~0.0 = disabled). has_value_embed and the
    resid/x0-lambda init values are already-decided, concrete choices made once outside modelcore
    when a config tree is materialized -- this module has no policy of its own about which layers
    get which; it only applies whatever it's given."""
    PARAM_ROLES = {"resid_lambda": "resid_scalar", "x0_lambda": "x0_scalar"}

    def __init__(self, n_embd, n_head, n_kv_head, layer_idx, window, rope, padded_vocab_size,
                 resid_lambda_init, x0_lambda_init, has_value_embed, runtime=None):
        super().__init__()
        self.attn = CausalSelfAttention(n_embd, n_head, n_kv_head, layer_idx, window, rope, padded_vocab_size,
                                         has_value_embed, runtime=runtime)
        self.mlp = MLP(n_embd)
        self.resid_lambda = nn.Parameter(torch.empty(()))  # fake init, real init in init_weights()
        self.x0_lambda = nn.Parameter(torch.empty(()))     # fake init, real init in init_weights()
        self._resid_lambda_init = resid_lambda_init
        self._x0_lambda_init = x0_lambda_init

    @torch.no_grad()
    def init_weights(self):
        self.attn.init_weights()
        self.mlp.init_weights()
        self.resid_lambda.fill_(self._resid_lambda_init)
        self.x0_lambda.fill_(self._x0_lambda_init)

    def layer_spec(self):
        return self.attn.layer_spec()

    def forward(self, x, x0, idx, kv_cache, kv_bus=None, doc_args=None):
        x = self.resid_lambda * x + self.x0_lambda * x0
        x = x + self.attn(norm(x), idx, kv_cache, kv_bus, doc_args)
        x = x + self.mlp(norm(x))
        return x


@register_component("plain_block", needs=("n_embd", "padded_vocab_size", "rope", "runtime"), validate=_validate_plain_block)
class PlainBlock(BaseBlock):
    """Plain pre-norm residual block: x = x + attn(norm(x)); x = x + mlp(norm(x)). No per-layer
    resid/x0-lambda mixing, no value embeddings, no smear/backout -- unlike Block above, this is a
    deliberately boring baseline. Reuses CausalSelfAttention unmodified (GQA/RoPE/QK-norm are not
    architecture-specific tricks); only the MLP (SwiGLU) and the absence of lambda mixing differ.
    kv_slot/produces_kv pass straight through to CausalSelfAttention (both default to today's
    one-slot-per-layer behavior) so this block also serves cross-layer KV sharing without a fork.

    No PARAM_ROLES declaration needed: attn's matrices default to role "matrix" via Linear, and
    mlp (SwiGLUMLP) is the same. A KV-sharing consumer block (produces_kv=False) simply has fewer
    Linear submodules -- nothing to declare either way."""

    def __init__(self, n_embd, n_head, n_kv_head, layer_idx, window, rope, padded_vocab_size,
                 kv_slot=None, produces_kv=True, runtime=None):
        super().__init__()
        self.attn = CausalSelfAttention(n_embd, n_head, n_kv_head, layer_idx, window, rope, padded_vocab_size,
                                         has_value_embed=False, kv_slot=kv_slot, produces_kv=produces_kv, runtime=runtime)
        self.mlp = SwiGLUMLP(n_embd)

    @torch.no_grad()
    def init_weights(self):
        self.attn.init_weights()
        self.mlp.init_weights()

    def layer_spec(self):
        return self.attn.layer_spec()

    def forward(self, x, x0, idx, kv_cache, kv_bus=None, doc_args=None):
        # x0 is part of the BaseBlock contract (see modelcore/components/contracts.py) but unused
        # here -- this topology has no x0 residual.
        x = x + self.attn(norm(x), idx, kv_cache, kv_bus, doc_args)
        x = x + self.mlp(norm(x))
        return x
