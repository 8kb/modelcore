import torch

from modelcore.catalog import register_component
from modelcore.components.contracts import BaseMixer, FeatureHost
from modelcore.components.linear import Linear
from modelcore.config.spec import AttentionLayerSpec, ComponentSpec
from modelcore.kernels.flash_attn import flash_attn
from modelcore.runtime import DEFAULT_RUNTIME


def _validate_attention(params, ctx):
    """Semantic checks for the attention mixer: n_embd/n_head/n_kv_head must be mutually
    consistent (the same constraints CausalSelfAttention asserts at construction time -- reported
    here as validation errors instead of crashing the build), window must be a real window value,
    and its features must fit its geometry. head_dim=None (derive from n_embd // n_head) still
    requires that division to be exact; an explicit head_dim decouples the two, so it isn't checked
    against n_embd/n_head here -- see validate._validate_kv_layout for the uniform-head_dim check
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
    produces_kv = params.get("produces_kv", True)
    if not produces_kv and params.get("kv_slot") is None:
        errors.append("produces_kv=False requires an explicit kv_slot pointing at the producer layer")
    if head_dim is None and n_head and n_embd is not None and n_embd % n_head == 0:
        head_dim = n_embd // n_head
    for feature in params.get("features") or []:
        if not isinstance(feature, ComponentSpec):
            continue
        if feature.type == "value_embed" and not produces_kv:
            errors.append("a KV-sharing consumer layer (produces_kv=False) cannot have a value_embed feature")
        if feature.type == "output_gate" and feature.params.get("granularity") == "block":
            block_size = feature.params.get("block_size")
            if isinstance(block_size, int) and isinstance(head_dim, int) and block_size > 0 and head_dim % block_size != 0:
                errors.append(f"head_dim ({head_dim}) must be divisible by output_gate.block_size ({block_size})")
    return errors


@register_component("attention", needs=("n_embd", "rope", "norm", "runtime"), validate=_validate_attention)
class CausalSelfAttention(BaseMixer, FeatureHost):
    """GQA + RoPE + QK-norm + FA3/SDPA sliding-window attention, as a Block's mixer. Owns its
    window and its layer_spec(); takes explicit dims rather than a config object.

    Features (see modelcore.components.features) attach at two hook points: `values` (after V is
    projected -- value_embed, the ResFormer-style value residual) and `output` (the per-head
    attention output, before c_proj -- output_gate). Which layers carry which feature is an
    architecture-level decision made once, outside modelcore, when a config tree is materialized.

    head_dim=None derives the per-head width as n_embd // n_head (still requiring n_embd % n_head
    == 0); an explicit value decouples attention width from n_head entirely -- c_q/c_k/c_v/c_proj
    size off n_head * head_dim, which need not equal n_embd, so adding heads at a fixed head_dim is
    not free the way it is when head_dim is derived. shared.rope's own head_dim is stated
    independently and is never cross-checked against this one -- keep them equal by hand, a
    mismatch is a runtime shape error, not a validation one.

    Cross-layer KV sharing: a layer built with produces_kv=False has no c_k/c_v at all and, at
    forward time, reads an earlier layer's already-RoPE'd/normed/scaled K/V out of kv_bus instead
    of computing its own -- it only projects and rotates its own queries. kv_slot identifies which
    KVCache slot this layer's K/V lives in; kv_slot=None means "my own slot at my layer_idx"
    (bound by the owning Block via bind_layer), and a consumer layer is given the producer's
    kv_slot explicitly. In a model where not every layer is attention, layer_idx is not a
    contiguous slot number, so such a config states every kv_slot explicitly. Passing the
    producer's own k/v tensors back into flash_attn_with_kvcache for the consumer (rather than
    k=None) sidesteps a real FA3-vs-SDPA divergence in what k=None means (see
    modelcore/docs/architecture.md's "Cross-layer KV sharing") -- the write is a no-op since the
    producer already wrote those exact tensors to that slot earlier in the same forward pass.

    Intra-document masking: doc_args (see modelcore.kernels.flash_attn.build_doc_args), when given,
    restricts attention to within each packed row's own document -- forwarded to
    flash_attn.flash_attn_func unchanged, training only (always None when kv_cache is not None;
    see Model.forward). This module doesn't derive it from idx itself: doc_args is per-batch
    runtime data built once outside torch.compile and threaded through like kv_bus."""
    HOOK_POINTS = ("values", "output")
    HAS_KV_SLOT = True  # what validate._validate_kv_layout looks for in a block's mixer

    def __init__(self, n_embd, n_head, n_kv_head, head_dim, window, kv_slot, produces_kv, features,
                 rope, norm, runtime=None):
        super().__init__()
        self.runtime = runtime or DEFAULT_RUNTIME
        self._kv_slot = kv_slot
        self.kv_slot = kv_slot  # resolved against layer_idx in bind_layer
        self.produces_kv = produces_kv
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.n_embd = n_embd
        # head_dim=None means "derive it from n_embd // n_head"; an explicit value decouples
        # attention width from n_embd entirely -- c_q's fan-out becomes n_head * head_dim, which
        # need not equal n_embd, and c_proj's fan-in follows it back.
        if head_dim is None:
            assert n_embd % n_head == 0
            head_dim = n_embd // n_head
        self.head_dim = head_dim
        assert n_kv_head <= n_head and n_head % n_kv_head == 0
        self.window = window
        self.rope = rope
        self.norm = norm  # shared config-selected norm, used for QK-norm
        self.c_q = Linear(n_embd, n_head * self.head_dim, bias=False)
        self.c_k = Linear(n_embd, n_kv_head * self.head_dim, bias=False) if produces_kv else None
        self.c_v = Linear(n_embd, n_kv_head * self.head_dim, bias=False) if produces_kv else None
        self.c_proj = Linear(n_head * self.head_dim, n_embd, bias=False)
        self._attach_features(features)

    def bind_layer(self, layer_idx):
        self.layer_idx = layer_idx
        self.kv_slot = layer_idx if self._kv_slot is None else self._kv_slot

    @torch.no_grad()
    def init_weights(self):
        s = 3**0.5 * self.n_embd**-0.5  # sqrt(3) multiplier makes sure Uniform achieves the same std as Normal
        torch.nn.init.uniform_(self.c_q.weight, -s, s)  # weights use Uniform to avoid outliers
        if self.produces_kv:
            torch.nn.init.uniform_(self.c_k.weight, -s, s)
            torch.nn.init.uniform_(self.c_v.weight, -s, s)
        torch.nn.init.zeros_(self.c_proj.weight)  # projections are zero
        self._init_features()

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
            v = self._hook("values", v, x, idx)

            # Apply Rotary Embeddings to queries and keys to get relative positional encoding
            q, k = self.rope(q, k, kv_cache)
            q, k = self.norm(q), self.norm(k)  # QK norm
            q = q * 1.2  # sharper attention (scale split between Q and K); part of the named architecture
            k = k * 1.2

            if kv_bus is not None:
                kv_bus[self.kv_slot] = (k, v)
        else:
            # Cross-layer KV sharing: reuse an earlier layer's K/V from this same forward pass --
            # it was already RoPE'd/normed/scaled by the producer, so only q needs that treatment.
            k, v = kv_bus[self.kv_slot]
            q = self.rope.apply_to_q(q, kv_cache)
            q = self.norm(q)
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

        y = self._hook("output", y, x)

        # Re-assemble the heads and project back to residual stream
        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y
