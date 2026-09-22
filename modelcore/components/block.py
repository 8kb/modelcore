import torch
import torch.nn as nn

from modelcore.catalog import register_component
from modelcore.components.attention import CausalSelfAttention
from modelcore.components.contracts import BaseBlock


def _validate_attention_shape(params, ctx):
    """Shared semantic checks for any attention-shaped block: n_embd/n_head/n_kv_head must be
    mutually consistent (the same constraints modelcore.components.attention.CausalSelfAttention
    asserts at construction time -- reported here as validation errors instead of crashing the
    build), and window must be a real window value. head_dim=None (derive from n_embd // n_head)
    still requires that division to be exact; an explicit head_dim decouples the two, so it isn't
    checked against n_embd/n_head here -- see _validate_kv_layout for the uniform-head_dim check
    the KV cache actually needs."""
    errors = []
    n_head = params.get("n_head")
    n_kv_head = params.get("n_kv_head", n_head)
    n_embd = ctx.get("n_embd")
    head_dim = params.get("head_dim")
    if head_dim is None and n_head and n_embd is not None and n_embd % n_head != 0:
        errors.append(f"n_embd ({n_embd}) must be divisible by n_head ({n_head})")
    if head_dim is not None and (not isinstance(head_dim, int) or isinstance(head_dim, bool) or head_dim <= 0):
        errors.append(f"head_dim must be a positive integer or null, got {head_dim!r}")
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


@register_component("gpt_block", needs=("n_embd", "padded_vocab_size", "rope", "norm", "runtime"), validate=_validate_gpt_block)
class Block(BaseBlock):
    """CausalSelfAttention + an MLP, plus the per-layer resid/x0-lambda residual mixing (inspired by
    modded-nanogpt): resid_lambda scales the residual stream at this layer (init ~1.0 = neutral),
    x0_lambda blends the initial embedding back in (init ~0.0 = disabled). has_value_embed, the
    resid/x0-lambda init values, and the `mlp` (a nested component spec: which nonlinearity, what
    inner width) are already-decided, concrete choices made once outside modelcore when a config
    tree is materialized -- this module has no policy of its own about which layers get which; it
    only applies whatever it's given. The norm is the config's shared one (`shared.norm`)."""
    PARAM_ROLES = {"resid_lambda": "resid_scalar", "x0_lambda": "x0_scalar"}

    def __init__(self, n_embd, n_head, n_kv_head, layer_idx, window, rope, norm, padded_vocab_size,
                 resid_lambda_init, x0_lambda_init, has_value_embed, mlp, head_dim, runtime=None):
        super().__init__()
        self.attn = CausalSelfAttention(n_embd, n_head, n_kv_head, layer_idx, window, rope, norm, padded_vocab_size,
                                         has_value_embed, head_dim, runtime=runtime)
        self.norm = norm
        self.mlp = mlp
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
        x = x + self.attn(self.norm(x), idx, kv_cache, kv_bus, doc_args)
        x = x + self.mlp(self.norm(x))
        return x


@register_component("plain_block", needs=("n_embd", "padded_vocab_size", "rope", "norm", "runtime"), validate=_validate_plain_block)
class PlainBlock(BaseBlock):
    """Plain pre-norm residual block: x = x + attn(norm(x)); x = x + mlp(norm(x)). No per-layer
    resid/x0-lambda mixing, no value embeddings, no smear/backout -- unlike Block above, this is a
    deliberately boring baseline. Reuses CausalSelfAttention unmodified (GQA/RoPE/QK-norm are not
    architecture-specific tricks); only the absence of lambda mixing differs -- the MLP is whatever
    the config's nested `mlp` spec says, exactly as for Block.
    kv_slot/produces_kv are stated explicitly (kv_slot=None means "own slot at my layer_idx") and
    pass straight through to CausalSelfAttention, so this block also serves cross-layer KV sharing
    without a fork.

    No PARAM_ROLES declaration needed: attn's matrices default to role "matrix" via Linear, and
    the mlp components are the same. A KV-sharing consumer block (produces_kv=False) simply has
    fewer Linear submodules -- nothing to declare either way."""

    def __init__(self, n_embd, n_head, n_kv_head, layer_idx, window, rope, norm, padded_vocab_size,
                 kv_slot, produces_kv, mlp, head_dim, runtime=None):
        super().__init__()
        self.attn = CausalSelfAttention(n_embd, n_head, n_kv_head, layer_idx, window, rope, norm, padded_vocab_size,
                                         has_value_embed=False, head_dim=head_dim, kv_slot=kv_slot,
                                         produces_kv=produces_kv, runtime=runtime)
        self.norm = norm
        self.mlp = mlp

    @torch.no_grad()
    def init_weights(self):
        self.attn.init_weights()
        self.mlp.init_weights()

    def layer_spec(self):
        return self.attn.layer_spec()

    def forward(self, x, x0, idx, kv_cache, kv_bus=None, doc_args=None):
        # x0 is part of the BaseBlock contract (see modelcore/components/contracts.py) but unused
        # here -- this topology has no x0 residual.
        x = x + self.attn(self.norm(x), idx, kv_cache, kv_bus, doc_args)
        x = x + self.mlp(self.norm(x))
        return x
