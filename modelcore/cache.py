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
        # Extra per-model state that isn't shaped like a k/v tensor (e.g. GPT's "smear" reads/
        # writes state["prev_embedding"]). Generic so architectures can stash whatever they need
        # without KVCache knowing about any one architecture.
        self.state = {}

    def reset(self):
        """Reset cache to empty state."""
        self.cache_seqlens.zero_()
        self.state = {}

    def get_pos(self):
        """Get current position (assumes all batch elements at same position)."""
        return self.cache_seqlens[0].item()

    def get_slot_cache(self, slot):
        """Return (k_cache, v_cache) views for a specific KV slot."""
        return self.k_cache[slot], self.v_cache[slot]

    def advance(self, num_tokens):
        """Advance the cache position by num_tokens."""
        self.cache_seqlens += num_tokens

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
        # Expand any batch=1 extra state (e.g. GPT's smear prev_embedding) to num_samples rows
        for key, value in other.state.items():
            self.state[key] = value.expand(self.batch_size, -1, -1).clone()
