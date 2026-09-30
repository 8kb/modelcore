"""
Generic FLOPs / parameter / KV-cache-bytes accounting, expressed purely in terms of
modelcore.config.spec.AttentionLayerSpec, plus ModelStats -- the frozen snapshot
ModelManager.stats(config) returns. None of this needs real weights: everything here is a
function of shapes only, which is why ModelManager.stats() can compute it from a meta-device
model without ever allocating real storage.

Ref: https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4
Ref: https://arxiv.org/abs/2204.02311 (PaLM paper)
Ref: https://arxiv.org/abs/2203.15556 (Chinchilla paper)
"""
from dataclasses import dataclass, field

from modelcore.components.linear import Linear
from modelcore.config.spec import AttentionLayerSpec, RecurrentLayerSpec


def num_matmul_params(model) -> int:
    """The number of parameters that participate in matmuls with the token stream, i.e.
    contribute 2 FLOPs/param to the forward pass. Counted structurally: every matmul in a
    modelcore component goes through the Linear class, while non-matmul params (embeddings =
    lookups, per-layer scalars) are nn.Embedding or raw Parameters."""
    return sum(m.weight.numel() for m in model.modules() if isinstance(m, Linear))


def _attention(layer_specs):
    """(layer index, spec) for every attention layer. layer_specs has one entry per block -- an
    AttentionLayerSpec, a RecurrentLayerSpec (a mixer with a fixed-size state instead of a KV
    cache), or None -- and the index into it is what a default kv_slot means, so it is kept."""
    return [(i, s) for i, s in enumerate(layer_specs) if isinstance(s, AttentionLayerSpec)]


def _effective_window(window, cap):
    """window=-1 means unlimited/full context, capped at `cap` (sequence_len or context_len)."""
    return cap if window < 0 else min(window, cap)


def estimate_flops(layer_specs, matmul_params, sequence_len, extra_fwd_flops=0) -> int:
    """FLOPs per token for the model (forward + backward). Each matmul weight parameter
    contributes 2 FLOPs (multiply, accumulate) in forward, 4x that in backward => 6x total. On
    top of that, 12 * h * q * effective_seq_len accounts for the key @ query matmul inside
    attention; with sliding windows, effective_seq_len varies per layer (capped by window size).
    This is ~1% off the exact Chinchilla-paper formula (which also counts the embedding lookup
    and softmax exp/sum/divide as FLOPs; both ignored here). extra_fwd_flops is the forward FLOPs
    per token of everything that is neither a Linear matmul nor attention (a conv's taps, a scan --
    see RecurrentLayerSpec.fwd_flops_per_token and BaseFeature.fwd_flops_per_token); training pays
    3x that (forward + a 2x backward)."""
    attn_flops = 0
    for _, spec in _attention(layer_specs):
        effective_seq = _effective_window(spec.window, sequence_len)
        attn_flops += 12 * spec.n_head * spec.head_dim * effective_seq
    return 6 * matmul_params + attn_flops + 3 * extra_fwd_flops


def estimate_decode_flops(layer_specs, matmul_params, context_len, extra_fwd_flops=0) -> int:
    """Forward FLOPs to decode one token at a given context length during inference: 2 FLOPs per
    matmul param, plus attention over min(context, window) per layer, plus extra_fwd_flops (the
    same per-token cost whatever the context length -- a recurrent layer's state is constant)."""
    attn_flops = 0
    for _, spec in _attention(layer_specs):
        w = _effective_window(spec.window, context_len)
        attn_flops += 4 * spec.n_head * spec.head_dim * w
    return 2 * matmul_params + attn_flops + extra_fwd_flops


def estimate_prefill_flops(layer_specs, matmul_params, num_tokens, extra_fwd_flops=0) -> int:
    """Forward FLOPs to prefill a prompt: causal, so token t attends to min(t, window)."""
    attn_flops = 0
    for _, spec in _attention(layer_specs):
        w = _effective_window(spec.window, num_tokens)
        attended_tokens = w * (w + 1) // 2 + (num_tokens - w) * w  # ramp up to w, then flat
        attn_flops += 4 * spec.n_head * spec.head_dim * attended_tokens
    return 2 * matmul_params * num_tokens + attn_flops + extra_fwd_flops * num_tokens


def distinct_kv_specs(layer_specs):
    """One spec per distinct KV cache slot (see AttentionLayerSpec.kv_slot), rather than one per
    layer -- layers that share a slot (cross-layer KV sharing) must not be double-counted when
    accounting for what's actually *stored*. Layer order is preserved; a layer with kv_slot=None
    is its own slot at its own position, matching kv_cache_spec()'s convention below."""
    seen = set()
    distinct = []
    for i, spec in _attention(layer_specs):
        slot = i if spec.kv_slot is None else spec.kv_slot
        if slot not in seen:
            seen.add(slot)
            distinct.append(spec)
    return distinct


def kv_bytes_per_token(layer_specs, dtype_itemsize) -> int:
    """Bytes to *store* one token of KV cache during inference, per row -- one contribution per
    distinct KV slot, not per layer, so cross-layer KV sharing correctly shows a smaller footprint
    than one-slot-per-layer at the same layer count."""
    return sum(2 * spec.n_kv_head * spec.head_dim * dtype_itemsize for spec in distinct_kv_specs(layer_specs))


def kv_read_bytes(layer_specs, dtype_itemsize, context_len) -> int:
    """Bytes of KV cache *read* by one decode step at a given context length, per row. Summed per
    layer (not per slot): a layer that reuses another layer's stored K/V still issues its own read
    of that cache during its own attention call. Sliding-window layers only read the last `window`
    tokens."""
    total = 0
    for _, spec in _attention(layer_specs):
        w = _effective_window(spec.window, context_len)
        total += 2 * spec.n_kv_head * spec.head_dim * dtype_itemsize * w
    return total


def kv_cache_spec(layer_specs) -> dict:
    """What modelcore.cache.KVCache needs to allocate: num_kv_slots, num_heads, head_dim.
    num_kv_slots is the number of *distinct* KV caches, which can be fewer than len(layer_specs)
    when layers share a slot (see AttentionLayerSpec.kv_slot). Requires uniform n_kv_head/head_dim
    across layers -- a genuinely heterogeneous-KV architecture would need KVCache itself
    generalized, not just this function. Only attention layers count; a model with none (pure
    SSM/conv) gets num_kv_slots=0 and zero-sized k/v tensors -- its per-row state lives in
    KVCache.state, see recurrent_state_elems."""
    assert layer_specs, "layer_specs is empty"
    attn = _attention(layer_specs)
    if not attn:
        return {"num_heads": 0, "head_dim": 0, "num_kv_slots": 0}
    n_kv_heads = {s.n_kv_head for _, s in attn}
    head_dims = {s.head_dim for _, s in attn}
    assert len(n_kv_heads) == 1 and len(head_dims) == 1, (
        "kv_cache_spec() requires uniform n_kv_head/head_dim across layers"
    )
    slots = {i if s.kv_slot is None else s.kv_slot for i, s in attn}
    assert slots == set(range(len(slots))), (
        "kv_cache_spec() requires kv slots to be a contiguous 0..M-1 range"
    )
    return {"num_heads": attn[0][1].n_kv_head, "head_dim": attn[0][1].head_dim, "num_kv_slots": len(slots)}


def feature_costs(model) -> tuple[int, int]:
    """(per-row state elements, forward FLOPs per token) summed over every feature in the model."""
    from modelcore.components.contracts import BaseFeature
    features = [m for m in model.modules() if isinstance(m, BaseFeature)]
    return sum(f.state_elems() for f in features), sum(f.fwd_flops_per_token() for f in features)


def recurrent_fwd_flops(layer_specs) -> int:
    return sum(s.fwd_flops_per_token for s in layer_specs if isinstance(s, RecurrentLayerSpec))


def recurrent_state_elems(layer_specs, feature_state_elems=0) -> int:
    """Elements of per-row inference state held by recurrent mixers (plus features that keep state,
    e.g. canon) -- constant however long the context, where attention's KV grows per token."""
    return sum(s.state_elems for s in layer_specs if isinstance(s, RecurrentLayerSpec)) + feature_state_elems


def shape_summary(config, layer_specs) -> dict:
    """n_layer/n_embd/n_head/n_kv_head/sequence_len/window_pattern, reporting "mixed" wherever
    layers disagree -- a materialized tree's per-layer choices can vary by construction, so this
    is the one implementation every config gets (a uniform tree just degenerates to a single
    value everywhere, rather than needing a separate "flat config" code path)."""
    attn = [s for _, s in _attention(layer_specs)]
    n_heads = {s.n_head for s in attn}
    n_kv_heads = {s.n_kv_head for s in attn}
    windows = {s.window for s in attn}
    one = lambda values: next(iter(values)) if len(values) == 1 else ("mixed" if values else None)
    return {
        "n_layer": config.n_layer, "n_embd": config.n_embd,
        "n_head": one(n_heads), "n_kv_head": one(n_kv_heads),
        "sequence_len": config.sequence_len,
        "window_pattern": one(windows),
    }


def has_sliding_window(layer_specs, sequence_len) -> bool:
    return any(0 <= s.window < sequence_len for _, s in _attention(layer_specs))


@dataclass(frozen=True)
class ModelStats:
    """Frozen snapshot ModelManager.stats(config) returns -- everything about a config's shape,
    parameter counts, and cost that doesn't need real weights. params_by_role uses modelcore's
    generic role names (see modelcore.roles); num_scaling_params is the matrix+unembedding
    convention that gives the cleanest scaling laws (see dev/LOG.md Jan 27, 2026 for the original
    finding) -- callers wanting a different combination can read params_by_role directly."""
    n_layer: int
    params_by_role: dict
    num_params: int
    num_matmul_params: int
    layer_specs: list
    kv_cache_spec: dict
    shape_summary: dict
    flops_per_token: int
    has_sliding_window: bool
    _kv_dtype_itemsize: int = field(repr=False, default=2)
    state_elems_per_row: int = 0       # recurrent mixers' + stateful features' per-row cache elements
    extra_fwd_flops_per_token: int = 0  # non-matmul, non-attention forward FLOPs per token

    @property
    def num_scaling_params(self) -> int:
        return self.params_by_role.get("matrix", 0) + self.params_by_role.get("unembedding", 0)

    def decode_flops(self, context_len: int) -> int:
        return estimate_decode_flops(self.layer_specs, self.num_matmul_params, context_len,
                                     self.extra_fwd_flops_per_token)

    def prefill_flops(self, num_tokens: int) -> int:
        return estimate_prefill_flops(self.layer_specs, self.num_matmul_params, num_tokens,
                                      self.extra_fwd_flops_per_token)

    def state_bytes_per_row(self) -> int:
        """Bytes of fixed-size recurrent state per row (0 for a pure-attention model) -- the
        constant counterpart of kv_bytes_per_token, which grows with context."""
        return self.state_elems_per_row * self._kv_dtype_itemsize

    def kv_bytes_per_token(self) -> int:
        return kv_bytes_per_token(self.layer_specs, self._kv_dtype_itemsize)

    def kv_read_bytes(self, context_len: int) -> int:
        return kv_read_bytes(self.layer_specs, self._kv_dtype_itemsize, context_len)
