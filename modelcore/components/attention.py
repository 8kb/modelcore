import torch
import torch.nn as nn

from modelcore.components.linear import Linear
from modelcore.components.norm import norm
from modelcore.config.spec import AttentionLayerSpec
from modelcore.kernels.flash_attn import flash_attn
from modelcore.runtime import DEFAULT_RUNTIME


class CausalSelfAttention(nn.Module):
    """GQA + RoPE + QK-norm + optional value-residual (ResFormer-style) + FA3/SDPA sliding-window
    attention. Owns its window, its (optional) value-embedding table, and its layer_spec(); takes
    explicit dims rather than a config object so it can be reused by a block with different config
    field names. has_value_embed is a plain, already-decided boolean here -- which layers get a
    value embedding is an architecture-level policy decision made once, outside modelcore, when a
    config tree is materialized (the host application's depth-dial layer decides this, e.g. via a
    has_value_embed-parity rule); this module has no opinion about how that policy is chosen.

    Cross-layer KV sharing: a layer built with produces_kv=False has no c_k/c_v at all and, at
    forward time, reads an earlier layer's already-RoPE'd/normed/scaled K/V out of kv_bus instead
    of computing its own -- it only projects and rotates its own queries. kv_slot identifies which
    KVCache slot this layer's K/V lives in; layers that produce their own K/V default kv_slot to
    their layer_idx (today's one-slot-per-layer behavior), and a consumer layer is given the
    producer's kv_slot explicitly by whatever built the tree. Passing the producer's own k/v
    tensors back into flash_attn_with_kvcache for the consumer (rather than k=None) sidesteps a
    real FA3-vs-SDPA divergence in what k=None means (see modelcore/docs/architecture.md's
    "Cross-layer KV sharing") -- the write is a no-op since the producer already wrote those exact
    tensors to that slot earlier in the same forward pass.

    Intra-document masking: doc_args (see modelcore.kernels.flash_attn.build_doc_args), when given,
    restricts attention to within each packed row's own document -- forwarded to
    flash_attn.flash_attn_func unchanged, training only (always None when kv_cache is not None;
    see Model.forward). This module doesn't derive it from idx itself: doc_args is per-batch
    runtime data built once outside torch.compile and threaded through like kv_bus, not a
    per-layer policy."""
    PARAM_ROLES = {"value_embed": "value_embedding"}

    def __init__(self, n_embd, n_head, n_kv_head, layer_idx, window, rope, padded_vocab_size, has_value_embed,
                 kv_slot=None, produces_kv=True, runtime=None):
        super().__init__()
        assert produces_kv or not has_value_embed, "a KV-sharing consumer layer cannot have its own value embedding"
        self.runtime = runtime or DEFAULT_RUNTIME
        self.layer_idx = layer_idx
        self.kv_slot = layer_idx if kv_slot is None else kv_slot
        self.produces_kv = produces_kv
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head
        assert n_embd % n_head == 0
        assert n_kv_head <= n_head and n_head % n_kv_head == 0
        self.window = window
        self.rope = rope
        self.c_q = Linear(n_embd, n_head * self.head_dim, bias=False)
        self.c_k = Linear(n_embd, n_kv_head * self.head_dim, bias=False) if produces_kv else None
        self.c_v = Linear(n_embd, n_kv_head * self.head_dim, bias=False) if produces_kv else None
        self.c_proj = Linear(n_embd, n_embd, bias=False)
        kv_dim = n_kv_head * self.head_dim
        self.value_embed = nn.Embedding(padded_vocab_size, kv_dim) if has_value_embed else None
        self.ve_gate_channels = 12
        self.ve_gate = Linear(self.ve_gate_channels, n_kv_head, bias=False) if has_value_embed else None

    @torch.no_grad()
    def init_weights(self):
        s = 3**0.5 * self.n_embd**-0.5  # sqrt(3) multiplier makes sure Uniform achieves the same std as Normal
        torch.nn.init.uniform_(self.c_q.weight, -s, s)  # weights use Uniform to avoid outliers
        if self.produces_kv:
            torch.nn.init.uniform_(self.c_k.weight, -s, s)
            torch.nn.init.uniform_(self.c_v.weight, -s, s)
        torch.nn.init.zeros_(self.c_proj.weight)  # projections are zero
        if self.value_embed is not None:
            torch.nn.init.uniform_(self.value_embed.weight, -s, s)  # init like c_v: uniform with same std
            if self.runtime.compute_dtype != torch.float16:
                self.value_embed.to(dtype=self.runtime.compute_dtype)
        if self.ve_gate is not None:
            # Gate weights init with small positive values so gates start slightly above neutral
            torch.nn.init.uniform_(self.ve_gate.weight, 0.0, 0.02)

    def layer_spec(self):
        return AttentionLayerSpec(n_head=self.n_head, n_kv_head=self.n_kv_head, head_dim=self.head_dim,
                                   window=self.window, kv_slot=self.kv_slot)

    def forward(self, x, idx, kv_cache, kv_bus=None, doc_args=None):
        B, T, C = x.size()

        # Project the input to get queries. Shape: (B, T, H, D) - FA3's native layout, no transpose needed!
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)

        if self.produces_kv:
            k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
            v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

            # Value residual (ResFormer): mix in value embedding with input-dependent gate per head
            if self.value_embed is not None:
                ve = self.value_embed(idx).to(x.dtype).view(B, T, self.n_kv_head, self.head_dim)
                gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))  # (B, T, n_kv_head), range (0, 3)
                v = v + gate.unsqueeze(-1) * ve

            # Apply Rotary Embeddings to queries and keys to get relative positional encoding
            q, k = self.rope(q, k, kv_cache)
            q, k = norm(q), norm(k)  # QK norm
            q = q * 1.2  # sharper attention (split scale between Q and K), TODO think through better
            k = k * 1.2

            if kv_bus is not None:
                kv_bus[self.kv_slot] = (k, v)
        else:
            # Cross-layer KV sharing: reuse an earlier layer's K/V from this same forward pass --
            # it was already RoPE'd/normed/scaled by the producer, so only q needs that treatment.
            k, v = kv_bus[self.kv_slot]
            q = self.rope.apply_to_q(q, kv_cache)
            q = norm(q)
            q = q * 1.2

        # Flash Attention (FA3 or SDPA fallback)
        # window_size is (left, right) tuple: (N, 0) for causal, (-1, 0) for full context
        window_size = (self.window, 0)
        if kv_cache is None:
            # Training: causal attention with optional sliding window and intra-document masking
            y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size, doc_args=doc_args)
        else:
            # Inference: use flash_attn_with_kvcache which handles cache management
            k_cache, v_cache = kv_cache.get_slot_cache(self.kv_slot)
            y = flash_attn.flash_attn_with_kvcache(
                q, k_cache, v_cache,
                k=k, v=v,
                cache_seqlens=kv_cache.cache_seqlens,
                causal=True,
                window_size=window_size,
            )

        # Re-assemble the heads and project back to residual stream
        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y
