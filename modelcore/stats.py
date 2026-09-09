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


def num_matmul_params(model) -> int:
    """The number of parameters that participate in matmuls with the token stream, i.e.
    contribute 2 FLOPs/param to the forward pass. Counted structurally: every matmul in a
    modelcore component goes through the Linear class, while non-matmul params (embeddings =
    lookups, per-layer scalars) are nn.Embedding or raw Parameters."""
    return sum(m.weight.numel() for m in model.modules() if isinstance(m, Linear))


def _effective_window(window, cap):
    """window=-1 means unlimited/full context, capped at `cap` (sequence_len or context_len)."""
    return cap if window < 0 else min(window, cap)


def estimate_flops(layer_specs, matmul_params, sequence_len) -> int:
    """FLOPs per token for the model (forward + backward). Each matmul weight parameter
    contributes 2 FLOPs (multiply, accumulate) in forward, 4x that in backward => 6x total. On
    top of that, 12 * h * q * effective_seq_len accounts for the key @ query matmul inside
    attention; with sliding windows, effective_seq_len varies per layer (capped by window size).
    This is ~1% off the exact Chinchilla-paper formula (which also counts the embedding lookup
    and softmax exp/sum/divide as FLOPs; both ignored here)."""
    attn_flops = 0
    for spec in layer_specs:
        effective_seq = _effective_window(spec.window, sequence_len)
        attn_flops += 12 * spec.n_head * spec.head_dim * effective_seq
    return 6 * matmul_params + attn_flops


def estimate_decode_flops(layer_specs, matmul_params, context_len) -> int:
    """Forward FLOPs to decode one token at a given context length during inference: 2 FLOPs per
    matmul param, plus attention over min(context, window) per layer."""
    attn_flops = 0
    for spec in layer_specs:
        w = _effective_window(spec.window, context_len)
        attn_flops += 4 * spec.n_head * spec.head_dim * w
    return 2 * matmul_params + attn_flops


def estimate_prefill_flops(layer_specs, matmul_params, num_tokens) -> int:
    """Forward FLOPs to prefill a prompt: causal, so token t attends to min(t, window)."""
    attn_flops = 0
    for spec in layer_specs:
        w = _effective_window(spec.window, num_tokens)
        attended_tokens = w * (w + 1) // 2 + (num_tokens - w) * w  # ramp up to w, then flat
        attn_flops += 4 * spec.n_head * spec.head_dim * attended_tokens
    return 2 * matmul_params * num_tokens + attn_flops


def distinct_kv_specs(layer_specs):
    """One spec per distinct KV cache slot (see AttentionLayerSpec.kv_slot), rather than one per
    layer -- layers that share a slot (cross-layer KV sharing) must not be double-counted when
    accounting for what's actually *stored*. Layer order is preserved; a layer with kv_slot=None
    is its own slot at its own position, matching kv_cache_spec()'s convention below."""
    seen = set()
    distinct = []
    for i, spec in enumerate(layer_specs):
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
    for spec in layer_specs:
        w = _effective_window(spec.window, context_len)
        total += 2 * spec.n_kv_head * spec.head_dim * dtype_itemsize * w
    return total


def kv_cache_spec(layer_specs) -> dict:
    """What modelcore.cache.KVCache needs to allocate: num_kv_slots, num_heads, head_dim.
    num_kv_slots is the number of *distinct* KV caches, which can be fewer than len(layer_specs)
    when layers share a slot (see AttentionLayerSpec.kv_slot). Requires uniform n_kv_head/head_dim
    across layers -- a genuinely heterogeneous-KV architecture would need KVCache itself
    generalized, not just this function."""
    assert layer_specs, "layer_specs is empty"
    n_kv_heads = {s.n_kv_head for s in layer_specs}
    head_dims = {s.head_dim for s in layer_specs}
    assert len(n_kv_heads) == 1 and len(head_dims) == 1, (
        "kv_cache_spec() requires uniform n_kv_head/head_dim across layers"
    )
    slots = {i if s.kv_slot is None else s.kv_slot for i, s in enumerate(layer_specs)}
    assert slots == set(range(len(slots))), (
        "kv_cache_spec() requires kv slots to be a contiguous 0..M-1 range"
    )
    return {"num_heads": layer_specs[0].n_kv_head, "head_dim": layer_specs[0].head_dim, "num_kv_slots": len(slots)}


def shape_summary(config, layer_specs) -> dict:
    """n_layer/n_embd/n_head/n_kv_head/sequence_len/window_pattern, reporting "mixed" wherever
    layers disagree -- a materialized tree's per-layer choices can vary by construction, so this
    is the one implementation every config gets (a uniform tree just degenerates to a single
    value everywhere, rather than needing a separate "flat config" code path)."""
    n_heads = {s.n_head for s in layer_specs}
    n_kv_heads = {s.n_kv_head for s in layer_specs}
    windows = {s.window for s in layer_specs}
    return {
        "n_layer": config.n_layer, "n_embd": config.n_embd,
        "n_head": next(iter(n_heads)) if len(n_heads) == 1 else "mixed",
        "n_kv_head": next(iter(n_kv_heads)) if len(n_kv_heads) == 1 else "mixed",
        "sequence_len": config.sequence_len,
        "window_pattern": next(iter(windows)) if len(windows) == 1 else "mixed",
    }


def has_sliding_window(layer_specs, sequence_len) -> bool:
    return any(0 <= s.window < sequence_len for s in layer_specs)


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

    @property
    def num_scaling_params(self) -> int:
        return self.params_by_role.get("matrix", 0) + self.params_by_role.get("unembedding", 0)

    def decode_flops(self, context_len: int) -> int:
        return estimate_decode_flops(self.layer_specs, self.num_matmul_params, context_len)

    def prefill_flops(self, num_tokens: int) -> int:
        return estimate_prefill_flops(self.layer_specs, self.num_matmul_params, num_tokens)

    def kv_bytes_per_token(self) -> int:
        return kv_bytes_per_token(self.layer_specs, self._kv_dtype_itemsize)

    def kv_read_bytes(self, context_len: int) -> int:
        return kv_read_bytes(self.layer_specs, self._kv_dtype_itemsize, context_len)
