"""
Causal depthwise convolution with carried state, and the gated short-conv mixer built on it.

causal_depthwise_conv is the one implementation every convolutional piece shares (the `short_conv`
mixer here, the `canon` feature in components/features.py), so "full sequence", "prefill with a
cache" and "one decode step" are the same arithmetic and the cached path can't drift from the
training path. It is written as K shifted multiply-adds rather than F.conv1d because K is tiny
(2-4), and because that form takes a per-token document mask for free.
"""
import torch
import torch.nn as nn

from modelcore.catalog import register_component
from modelcore.components.contracts import BaseMixer, is_int
from modelcore.components.linear import Linear
from modelcore.config.spec import RecurrentLayerSpec


def causal_depthwise_conv(x, weight, state=None, doc_ids=None):
    """y_t = sum_{j<K} weight[:, K-1-j] * x_{t-j}: a causal depthwise convolution.

    x (B, T, D); weight (D, K); state (B, K-1, D) = the K-1 inputs just before x (None = zeros,
    which is what training and a fresh cache mean by "before the sequence"). doc_ids (B, T), for
    intra-document masking: a tap only reads a token from the same document as the output token,
    so a packed row gives exactly what each document would on its own. Returns (y, new_state),
    new_state being the last K-1 inputs seen -- what the next call's `state` should be."""
    B, T, D = x.shape
    K = weight.size(1)
    if state is None:
        state = x.new_zeros(B, K - 1, D)
    xs = torch.cat([state.to(x.dtype), x], dim=1)  # (B, K-1+T, D)
    w = weight.to(x.dtype)
    y = None
    for j in range(K):  # tap j reads the input `lag` steps back
        term = w[:, j] * xs[:, j:j + T]
        lag = K - 1 - j
        if doc_ids is not None and lag > 0:
            same = torch.nn.functional.pad(doc_ids[:, lag:] == doc_ids[:, :-lag], (lag, 0), value=False)
            term = term * same.unsqueeze(-1).to(term.dtype)
        y = term if y is None else y + term
    return y, xs[:, xs.size(1) - (K - 1):]


def _validate_short_conv(params, ctx):
    k = params.get("kernel_size")
    if not is_int(k) or k < 1:
        return [f"kernel_size must be a positive integer, got {k!r}"]
    return []


@register_component("short_conv", needs=("n_embd",), validate=_validate_short_conv)
class ShortConv(BaseMixer):
    """Gated short convolution (LFM2-style): B, C, v = in_proj(x); y = B * v; a causal depthwise
    conv over y; out_proj(C * conv(y)). Mixes only kernel_size-1 tokens back, so it needs no
    positional encoding and no KV cache: its inference state is the last kernel_size-1 values of
    B*v per row, a constant however long the context. Intra-document masking (doc_args) keeps the
    conv from reaching across a packed row's document boundary.

    in_proj/out_proj are Linear (role matrix); the depthwise filter is role "conv" (AdamW, no
    weight decay -- a (n_embd, kernel_size) filter is the wrong shape for Muon)."""
    PARAM_ROLES = {"conv_weight": "conv"}

    def __init__(self, n_embd, kernel_size):
        super().__init__()
        self.n_embd = n_embd
        self.kernel_size = kernel_size
        self.in_proj = Linear(n_embd, 3 * n_embd, bias=False)
        self.conv_weight = nn.Parameter(torch.empty(n_embd, kernel_size))  # fake init, real in init_weights()
        self.out_proj = Linear(n_embd, n_embd, bias=False)

    def bind_layer(self, layer_idx):
        self.layer_idx = layer_idx

    @torch.no_grad()
    def init_weights(self):
        s = 3**0.5 * self.n_embd**-0.5  # same scheme as attention's projections
        torch.nn.init.uniform_(self.in_proj.weight, -s, s)
        bound = self.kernel_size**-0.5  # torch's own Conv1d default for a depthwise filter
        torch.nn.init.uniform_(self.conv_weight, -bound, bound)
        torch.nn.init.zeros_(self.out_proj.weight)  # projections are zero

    def layer_spec(self):
        D, K = self.n_embd, self.kernel_size
        return RecurrentLayerSpec("short_conv", state_elems=(K - 1) * D, fwd_flops_per_token=2 * K * D + 2 * D)

    def forward(self, x, idx, cache, bus=None, doc_args=None):
        b, c, v = self.in_proj(x).chunk(3, dim=-1)
        key = f"short_conv.{self.layer_idx}"
        state = None if cache is None else cache.state.get(key)
        doc_ids = doc_args.doc_ids if (doc_args is not None and cache is None) else None
        z, new_state = causal_depthwise_conv(b * v, self.conv_weight, state, doc_ids)
        if cache is not None:
            cache.state[key] = new_state
        return self.out_proj(c * z)
