import torch
import torch.nn as nn

from modelcore.catalog import register_component
from modelcore.components.rope import apply_rotary_emb, precompute_rotary_embeddings
from modelcore.runtime import DEFAULT_RUNTIME


@register_component("rotary", needs=("sequence_len", "runtime"))
class RotaryEmbedding(nn.Module):
    """Owns the cos/sin buffers and the position-encoding step for attention layers sharing one
    instance. Deliberately not baked into BaseBlock.forward's signature -- position encoding is
    an attention-internal concern an architecture can swap (RoPE / NoPE / ALiBi) without touching
    the block/trunk contract.

    NOTE meta-device footgun: __init__ may run under torch.device("meta") (see
    docs/architecture.md) -- the cos/sin buffers it registers here are placeholder shapes only;
    real values are computed in init_weights(). They are also persistent=False (never saved to a
    checkpoint), which is why ModelManager.load_model() calls init_weights() even when loading a
    checkpoint."""

    def __init__(self, head_dim, sequence_len, over_compute, runtime=None):
        super().__init__()
        self.head_dim = head_dim
        self.runtime = runtime or DEFAULT_RUNTIME
        # rotary embeddings are cheap in memory, so over-compute by over_compute x sequence_len
        # rather than growing the cache dynamically; forward() asserts we never exceed this.
        self.rotary_seq_len = sequence_len * over_compute
        # Respect whatever device is ambient at construction time (like nn.Linear/nn.Embedding
        # do): "meta" if built under torch.device("meta") (the usual case, via Model.__init__), a
        # real device otherwise -- so this module is directly usable standalone too, without
        # requiring an external to_empty(device) call before init_weights() is meaningful.
        device = torch.empty(0).device
        cos, sin = precompute_rotary_embeddings(self.rotary_seq_len, head_dim, device=device, dtype=self.runtime.compute_dtype)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        cos, sin = precompute_rotary_embeddings(self.rotary_seq_len, self.head_dim, device=self.cos.device, dtype=self.runtime.compute_dtype)
        self.cos, self.sin = cos, sin

    def _cos_sin(self, T, kv_cache):
        assert self.cos.dtype == self.runtime.compute_dtype, f"Rotary embeddings must be in {self.runtime.compute_dtype}, got {self.cos.dtype}"
        if kv_cache is None:
            T0 = 0
        else:
            # uniform_pos() is KVCache's ragged-aware accessor; a duck-typed cache offering only
            # get_pos() (the minimal interface) is uniform by definition.
            uniform_pos = getattr(kv_cache, "uniform_pos", None)
            T0 = uniform_pos() if uniform_pos is not None else kv_cache.get_pos()
        if T0 is not None:
            # Every row at the same position (training, naive generate, batch-1 or replicated
            # decode): one (1, T, 1, D/2) window, broadcast over rows by apply_rotary_emb.
            assert T0 + T <= self.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T0 + T} > {self.cos.size(1)}"
            return self.cos[:, T0:T0 + T], self.sin[:, T0:T0 + T]
        # Right-ragged cache (see KVCache.prefill_row): row i's next position is cache_seqlens[i],
        # so each row needs its own window. apply_rotary_emb broadcasts a (B, T, 1, D/2) cos over
        # q/k's (B, T, H, D/2) halves exactly as it does the uniform (1, T, 1, D/2) form.
        pos = kv_cache.cache_seqlens.to(torch.long)
        idx = pos[:, None] + torch.arange(T, device=pos.device)  # (B, T)
        assert int(idx.max().item()) < self.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {int(idx.max().item()) + 1} > {self.cos.size(1)}"
        B = pos.numel()
        flat = idx.reshape(-1)
        cos = self.cos[0].index_select(0, flat).view(B, T, 1, -1)
        sin = self.sin[0].index_select(0, flat).view(B, T, 1, -1)
        return cos, sin

    def forward(self, q, k, kv_cache):
        """Apply rotary position encoding to queries and keys, offsetting into the cache by the
        current KV-cache position (0 during training / naive generate)."""
        assert q.device == self.cos.device, f"Rotary embeddings and q are on different devices: {q.device} != {self.cos.device}"
        cos, sin = self._cos_sin(q.size(1), kv_cache)
        return apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)

    def apply_to_q(self, q, kv_cache):
        """Like forward, but for a KV-sharing consumer layer that only needs to rotate its own
        queries -- the K it reads from another layer's slot was already rotated by that layer."""
        assert q.device == self.cos.device, f"Rotary embeddings and q are on different devices: {q.device} != {self.cos.device}"
        cos, sin = self._cos_sin(q.size(1), kv_cache)
        return apply_rotary_emb(q, cos, sin)
