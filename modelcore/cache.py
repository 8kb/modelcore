"""
KVCache: the inference-time KV cache every modelcore Model reads/writes through. Designed for
Flash Attention 3's flash_attn_with_kvcache API.

Key differences from FA2-style cache:
- Tensors are (B, T, H, D) not (B, H, T, D)
- FA3 updates the cache in-place during flash_attn_with_kvcache
- Position tracked per batch element via cache_seqlens tensor
"""
import torch


class KVCache:
    def __init__(self, batch_size, num_heads, seq_len, head_dim, num_kv_slots, device, dtype):
        self.batch_size = batch_size
        self.max_seq_len = seq_len
        self.n_slots = num_kv_slots
        self.n_heads = num_heads
        self.head_dim = head_dim
        # Pre-allocate cache tensors: (n_slots, B, T, H, D). n_slots can be fewer than the
        # model's layer count when layers share a KV slot (cross-layer KV sharing).
        self.k_cache = torch.zeros(num_kv_slots, batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
        self.v_cache = torch.zeros(num_kv_slots, batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
        # Current sequence length per batch element (FA3 needs int32)
        self.cache_seqlens = torch.zeros(batch_size, dtype=torch.int32, device=device)
        # Host-side mirror of "are all rows at the same position?": the shared position, or None
        # once rows are right-ragged (see prefill_row). Kept without ever reading cache_seqlens
        # back to the host, which would be a device sync in the decode hot path. Only KVCache's
        # own methods may mutate cache_seqlens, or this goes stale.
        self._pos_uniform = 0
        # Extra per-model state that isn't shaped like a k/v tensor (e.g. GPT's "smear" reads/
        # writes state["prev_embedding"]). Generic so architectures can stash whatever they need
        # without KVCache knowing about any one architecture.
        self.state = {}

    def reset(self):
        """Reset cache to empty state. k_cache/v_cache are deliberately NOT zeroed: a row only
        reads keys at index <= its own current position, and every such index was written by that
        row in this generation (causal masking on both the FA3 and SDPA paths enforces it), so
        residual KV from an earlier use is unreachable rather than merely unlikely to matter."""
        self.cache_seqlens.zero_()
        self.state = {}
        self._pos_uniform = 0

    def get_pos(self):
        """The single position every batch element is at. Raises on a right-ragged cache (see
        prefill_row) rather than silently answering for row 0 -- use cache_seqlens/uniform_pos()."""
        assert self._pos_uniform is not None, \
            "get_pos() on a right-ragged KV cache; read cache_seqlens or check uniform_pos()"
        return self._pos_uniform

    def uniform_pos(self):
        """get_pos() without the assert: the shared position, or None if the cache is ragged."""
        return self._pos_uniform

    def get_slot_cache(self, slot):
        """Return (k_cache, v_cache) views for a specific KV slot."""
        return self.k_cache[slot], self.v_cache[slot]

    def advance(self, num_tokens):
        """Advance every row by num_tokens.

        Uniform by construction, and still correct for a right-ragged cache: decode steps every
        row exactly once per step (rows whose generation already finished included), so the
        *differences* between rows' positions -- which is all raggedness is -- never change. That
        makes this a load-bearing invariant, not an accident: anything that stops stepping a row
        (compacting finished rows out of the batch, per-row early exit) breaks ragged positions
        silently and must replace this method, not just be added beside it."""
        self.cache_seqlens += num_tokens
        if self._pos_uniform is not None:
            self._pos_uniform += num_tokens

    def prefill(self, other):
        """
        Copy cached KV from another cache into this one.
        Used when we do batch=1 prefill and then want to generate multiple samples in parallel.
        """
        assert self.get_pos() == 0, "Cannot prefill a non-empty KV cache"
        assert self.n_slots == other.n_slots and self.n_heads == other.n_heads and self.head_dim == other.head_dim
        assert self.max_seq_len >= other.max_seq_len
        other_pos = other.get_pos()
        self.k_cache[:, :, :other_pos, :, :] = other.k_cache[:, :, :other_pos, :, :]
        self.v_cache[:, :, :other_pos, :, :] = other.v_cache[:, :, :other_pos, :, :]
        self.cache_seqlens.fill_(other_pos)
        self._pos_uniform = other_pos
        # Expand any batch=1 extra state (e.g. GPT's smear prev_embedding) to num_samples rows
        for key, value in other.state.items():
            self.state[key] = value.expand(self.batch_size, -1, -1).clone()

    def prefill_row(self, row, other):
        """Copy a batch=1 prefilled cache into row `row`, leaving every other row alone -- the
        multi-prompt counterpart of prefill(), which replicates ONE prompt into every row. Rows
        filled this way end at different cache_seqlens: right-ragged, row i's KV occupying
        [0, cache_seqlens[i]) with no left padding and no position offset, which is what FA3's
        flash_attn_with_kvcache natively means. Fill one row per prompt, then decode all rows in
        lockstep. One row at a time on purpose, so a caller frees each transient batch=1 prefill
        cache before making the next."""
        assert other.batch_size == 1, "prefill_row copies from a batch=1 cache"
        assert 0 <= row < self.batch_size, f"row {row} out of range for batch_size {self.batch_size}"
        assert self.cache_seqlens[row].item() == 0, "Cannot prefill a non-empty KV cache row"
        assert self.n_slots == other.n_slots and self.n_heads == other.n_heads and self.head_dim == other.head_dim
        other_pos = other.get_pos()
        assert other_pos <= self.max_seq_len, f"prompt of {other_pos} tokens exceeds cache length {self.max_seq_len}"
        self.k_cache[:, row, :other_pos] = other.k_cache[:, 0, :other_pos]
        self.v_cache[:, row, :other_pos] = other.v_cache[:, 0, :other_pos]
        self.cache_seqlens[row] = other_pos
        # Per-row extra state (e.g. GPT's smear prev_embedding), allocated lazily from the source's
        # own shape so KVCache stays generic about what a model stashes here.
        for key, value in other.state.items():
            buf = self.state.get(key)
            if buf is None:
                buf = value.new_zeros((self.batch_size,) + tuple(value.shape[1:]))
                self.state[key] = buf
            buf[row] = value[0]
        # Uniform only if every row happens to be at the same place -- e.g. equal-length prompts.
        seqlens = self.cache_seqlens.tolist()
        self._pos_uniform = seqlens[0] if all(s == seqlens[0] for s in seqlens) else None
