"""
Unified Flash Attention interface with automatic FA3/SDPA switching.

Exports `flash_attn` module that matches the FA3 API exactly, but falls back
to PyTorch SDPA on incompatible CUDA GPUs, MPS, and CPU.

Usage (drop-in replacement for FA3):
    from modelcore.kernels.flash_attn import flash_attn

    # Training (no KV cache)
    y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)

    # Inference (with KV cache)
    y = flash_attn.flash_attn_with_kvcache(q, k_cache, v_cache, k=k, v=v, ...)

    # Training with intra-document masking (no cross-document attention within a packed row)
    doc_args = build_doc_args(idx, bos_token_id)  # outside any torch.compile region -- see below
    y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size, doc_args=doc_args)
"""
from typing import NamedTuple, Optional

import torch
import torch.nn.functional as F


# =============================================================================
# Detection: Try to load FA3 on CUDA GPUs
# =============================================================================
def _load_flash_attention_3():
    """Try to load Flash Attention 3. Returns (module_or_None, reason_string).

    The reason string is populated whenever module is None, so a caller printing a fallback
    warning can say *why* instead of just "not available" -- this used to be silently discarded
    (a bare `except Exception: return None`), which meant a real, possibly-fixable failure (HF
    hub unreachable, `kernels` import broken, an auth error, ...) was indistinguishable from a
    GPU that genuinely doesn't have a published kernel build. Found the hard way: a real
    4x A100-SXM4-80GB run fell back to SDPA with zero indication of why, even though the code's
    own compatibility comment (and the live HF hub state) say sm80 should work.
    """
    if not torch.cuda.is_available():
        return None, "no CUDA device available"
    try:
        major, _ = torch.cuda.get_device_capability()
        # FA3 kernels are currently compiled for Hopper (sm90), Ada (sm89) and Ampere (sm80/sm86)
        # Blackwell (sm100) needs SDPA fallback until FA3 is recompiled or FA4 is released
        import os
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        from kernels import get_kernel, has_kernel
        # The varunneal kernel obtains better results for H100/Hopper
        if major == 9:
            hf_kernel = "varunneal/flash-attention-3"
            return get_kernel(hf_kernel).flash_attn_interface, None
        else:
            hf_kernel = "kernels-community/flash-attn3"
            if has_kernel(hf_kernel):
                return get_kernel(hf_kernel).flash_attn_interface, None
            else:
                return None, f"kernels.has_kernel('{hf_kernel}') returned False for this GPU/torch/CUDA build"

    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


_fa3, FA3_LOAD_ERROR = _load_flash_attention_3()
HAS_FA3 = _fa3 is not None

# Override for testing: set to 'fa3', 'sdpa', or None (auto)
_override_impl = None


def _resolve_use_fa3():
    """Decide once whether to use FA3, based on availability, override, and dtype."""
    if _override_impl == 'fa3':
        assert HAS_FA3, "Cannot override to FA3: not available on this hardware"
        return True
    if _override_impl == 'sdpa':
        return False
    if HAS_FA3:
        # FA3 Hopper kernels only support bf16 and fp8; fp16/fp32 must use SDPA fallback
        from modelcore.runtime import DEFAULT_RUNTIME
        if DEFAULT_RUNTIME.compute_dtype == torch.bfloat16:
            return True
        return False
    return False

USE_FA3 = _resolve_use_fa3()


# =============================================================================
# Intra-document masking: derive per-row document boundaries from BOS positions
# =============================================================================
class DocArgs(NamedTuple):
    """Per-batch document-boundary info for intra-document attention masking. Built once outside
    the compiled model region (see build_doc_args) and threaded through as opaque data -- nothing
    downstream re-derives it from idx, matching the kv_bus threading pattern in
    modelcore.components.attention.

    doc_ids: (B, T) int32, 0-based, incrementing at every document start within a row -- used by
        the SDPA fallback to build an explicit intra-document mask.
    cu_seqlens: (max_docs+1,) int32 cumulative sequence lengths over the flattened (B*T,) stream,
        fixed-shape (padded with the trailing total) so its shape never changes across steps --
        used by the FA3 varlen path. None when FA3 isn't the active backend (SDPA doesn't need it).
    max_seqlen: static python int, the padded length used to build cu_seqlens's fixed shape.
    """
    doc_ids: torch.Tensor
    cu_seqlens: Optional[torch.Tensor] = None
    max_seqlen: Optional[int] = None


DEFAULT_MAX_DOCS_PER_ROW = 64
"""Default cap on documents-per-row for build_doc_args's cu_seqlens (see there). ClimbMix at
sequence_len=2048 averages ~4.2 documents/row (3,727,360 documents / 893,729 sequences, measured
against a real prepared dataset) -- 64 is a >15x safety margin over that average, not a measured
per-row maximum. Pass max_docs explicitly for a dataset/sequence-length combination where that
margin doesn't hold (very short documents relative to sequence_len)."""


def build_doc_args(idx, bos_token_id, padding_id=None, max_docs=None):
    """Derive document boundaries from BOS positions in idx, (B, T) token ids.

    Call this OUTSIDE any torch.compile region and pass its result in as plain data:
    `nonzero()`-driven boundary detection inside a compiled model hits torch.compile's recompile
    limit, and a variable-shape cu_seqlens recompiles the graph on every step (25s/iter in
    upstream's own measurement -- see docs/upstream/LOG.md's "Varlen Attention" entry). Both are
    avoided here: doc_ids is a plain cumsum (no data-dependent shape), and cu_seqlens is padded to
    a fixed `max_docs` so its shape is constant regardless of how many documents actually occur.

    max_docs directly sizes the FA3 varlen kernel's backward-pass scratch allocation -- it is NOT
    just cu_seqlens's own (negligible) tensor size. The kernel treats cu_seqlens as declaring that
    many sequences regardless of how many are actually non-empty, and allocates workspace
    accordingly: defaulting this to the worst case (every token its own document, B*T) OOM'd a
    real H100 run trying to allocate 28GB of backward scratch for a declared batch of 131,072
    sequences when the real batch had ~270 documents. Default is `DEFAULT_MAX_DOCS_PER_ROW * B` --
    tune it down for less memory, up if a dataset genuinely packs more documents per row.

    A run of consecutive BOS ids (e.g. a pad tail written with bos_token_id as filler) collapses
    into one document rather than one document per pad token -- `is_start` only fires on the first
    BOS of a run.

    padding_id: what a packer used to fill unused row capacity (datacore.packing.BestFitPadPacker;
    irrelevant for a never-padded pretraining row). None (default) means "unknown, or the packer
    reused bos_token_id itself" (every dataset prepared before this parameter existed, and
    BestFitPadPacker's own default) -- in that case a document consisting ENTIRELY of
    bos_token_id can only be that pad tail, since a real document always has non-BOS content after
    its own leading BOS, so it's folded into the preceding document instead of counted as its own
    (a row that is 100% padding, with no preceding document to fold into, is left alone). When
    padding_id is a real, distinct value, no special-casing is needed at all: it never equals
    bos_token_id, so it never triggers is_start in the first place, and padded positions already
    inherit the preceding document's id from the cumsum below.
    """
    B, T = idx.shape
    is_bos = idx == bos_token_id
    is_start = is_bos.clone()
    is_start[:, 1:] &= ~is_bos[:, :-1]  # only the first BOS of a run starts a new document
    doc_ids = is_start.to(torch.int32).cumsum(dim=1) - 1  # 0-based within each row
    doc_ids = doc_ids.clamp(min=0)  # a row that starts mid-document (shouldn't happen -- every
                                     # row is BOS-aligned -- but stay defined rather than negative)

    if padding_id is None:
        last_doc = doc_ids.max(dim=1, keepdim=True).values  # (B, 1)
        is_last_doc = doc_ids == last_doc
        tail_is_all_bos = (is_bos | ~is_last_doc).all(dim=1, keepdim=True)  # (B, 1)
        should_fold = tail_is_all_bos & (last_doc > 0)  # nothing to fold into if it's the only doc
        fold_mask = is_last_doc & should_fold
        doc_ids = doc_ids - fold_mask.to(doc_ids.dtype)
        is_start = is_start & ~fold_mask  # the folded document's own leading BOS is no longer a
                                           # boundary either -- keeps the FA3 path (below) consistent

    if not USE_FA3:
        return DocArgs(doc_ids=doc_ids)

    # Segment starts over the flattened (B*T,) stream: every row start is a document start too
    # (every row is BOS-aligned), plus every in-row document start already marked by is_start.
    row_starts = torch.zeros_like(is_start)
    row_starts[:, 0] = True
    starts_flat = torch.nonzero((is_start | row_starts).view(-1), as_tuple=True)[0].to(torch.int32)
    # NOTE: nonzero() above is fine here -- build_doc_args always runs outside torch.compile.
    num_docs = starts_flat.numel()
    total = B * T
    cap = max_docs if max_docs is not None else DEFAULT_MAX_DOCS_PER_ROW * B
    assert num_docs <= cap, (
        f"{num_docs} document segments exceeds max_docs={cap} ({'explicit' if max_docs is not None else f'default: {DEFAULT_MAX_DOCS_PER_ROW} * batch_size={B}'}) "
        f"-- this batch packs more documents/row than the default margin assumes; pass a larger max_docs to build_doc_args"
    )
    cu_seqlens = torch.full((cap + 1,), total, dtype=torch.int32, device=idx.device)
    cu_seqlens[:num_docs] = starts_flat
    cu_seqlens[num_docs] = total
    # everything past num_docs is already `total` -- zero-length trailing segments, which FA3
    # accepts (same trick modded-nanogpt uses for its own fixed-shape cu_seqlens).
    return DocArgs(doc_ids=doc_ids, cu_seqlens=cu_seqlens, max_seqlen=T)


# =============================================================================
# SDPA helpers
# =============================================================================
def _sdpa_attention(q, k, v, window_size, enable_gqa, doc_ids=None):
    """
    SDPA attention with sliding window support.
    q, k, v are (B, H, T, D) format.
    """
    Tq = q.size(2)
    Tk = k.size(2)
    window = window_size[0]

    # Full context, same length, no document masking requested: the fused fast path
    if doc_ids is None:
        if (window < 0 or window >= Tq) and Tq == Tk:
            return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa)

        # Single token generation
        if Tq == 1:
            if window >= 0 and window < Tk:
                # window is "left" tokens we need to include (window + 1) keys total
                start = max(0, Tk - (window + 1))
                k = k[:, :, start:, :]
                v = v[:, :, start:, :]
            return F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=enable_gqa)

    # Need an explicit mask: sliding window/chunk inference, or intra-document masking
    device = q.device
    # For chunk inference (Tq != Tk), is_causal is not aligned to cache position => build an explicit bool mask
    row_idx = (Tk - Tq) + torch.arange(Tq, device=device).unsqueeze(1)
    col_idx = torch.arange(Tk, device=device).unsqueeze(0)
    mask = col_idx <= row_idx

    # sliding window (left)
    if window >= 0 and window < Tk:
        mask = mask & ((row_idx - col_idx) <= window)

    if doc_ids is not None:
        # doc_ids is (B, T); training-only path (Tq == Tk == T, no KV cache), so this aligns
        # directly with row_idx/col_idx without needing kv-cache-position bookkeeping.
        doc_mask = doc_ids[:, :, None] == doc_ids[:, None, :]  # (B, Tq, Tk)
        mask = mask.unsqueeze(0) & doc_mask
        mask = mask.unsqueeze(1)  # (B, 1, Tq, Tk) -- broadcasts across heads

    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=enable_gqa)

# =============================================================================
# Public API: Same interface as FA3
# =============================================================================
def flash_attn_func(q, k, v, causal=False, window_size=(-1, -1), doc_args=None):
    """
    Flash Attention for training (no KV cache).

    Args:
        q, k, v: Tensors of shape (B, T, H, D)
        causal: Whether to use causal masking
        window_size: (left, right) sliding window. -1 means unlimited.
        doc_args: optional DocArgs (see build_doc_args) restricting attention to within each
            packed row's own document. None (default) is exactly today's behavior.

    Returns:
        Output tensor of shape (B, T, H, D)
    """
    if USE_FA3:
        if doc_args is not None:
            B, T, H, D = q.shape
            out = _fa3.flash_attn_varlen_func(
                q.reshape(B * T, H, D), k.reshape(B * T, H, D), v.reshape(B * T, H, D),
                cu_seqlens_q=doc_args.cu_seqlens, cu_seqlens_k=doc_args.cu_seqlens,
                max_seqlen_q=doc_args.max_seqlen, max_seqlen_k=doc_args.max_seqlen,
                causal=causal, window_size=window_size,
            )
            return out.view(B, T, H, D)
        return _fa3.flash_attn_func(q, k, v, causal=causal, window_size=window_size)

    # SDPA fallback: transpose (B, T, H, D) -> (B, H, T, D)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    enable_gqa = q.size(1) != k.size(1)
    doc_ids = doc_args.doc_ids if doc_args is not None else None
    y = _sdpa_attention(q, k, v, window_size, enable_gqa, doc_ids=doc_ids)
    return y.transpose(1, 2)  # back to (B, T, H, D)


def flash_attn_with_kvcache(q, k_cache, v_cache, k=None, v=None, cache_seqlens=None,
                            causal=False, window_size=(-1, -1)):
    """
    Flash Attention with KV cache for inference.

    FA3 updates k_cache/v_cache in-place. Our SDPA fallback does the same.

    Args:
        q: Queries, shape (B, T_new, H, D)
        k_cache, v_cache: Pre-allocated cache tensors, shape (B, T_max, H_kv, D)
        k, v: New keys/values to insert, shape (B, T_new, H_kv, D)
        cache_seqlens: Current position in cache, shape (B,) int32
        causal: Whether to use causal masking
        window_size: (left, right) sliding window. -1 means unlimited.

    Returns:
        Output tensor of shape (B, T_new, H, D)
    """
    if USE_FA3:
        return _fa3.flash_attn_with_kvcache(
            q, k_cache, v_cache, k=k, v=v, cache_seqlens=cache_seqlens,
            causal=causal, window_size=window_size
        )

    # SDPA fallback: manually manage KV cache
    B, T_new, H, D = q.shape
    pos = cache_seqlens[0].item()  # assume uniform position across batch

    # Insert new k, v into cache (in-place, matching FA3 behavior)
    if k is not None and v is not None:
        k_cache[:, pos:pos+T_new, :, :] = k
        v_cache[:, pos:pos+T_new, :, :] = v

    # Get full cache up to current position + new tokens
    end_pos = pos + T_new
    k_full = k_cache[:, :end_pos, :, :]
    v_full = v_cache[:, :end_pos, :, :]

    # Transpose to SDPA layout: (B, T, H, D) -> (B, H, T, D)
    q_sdpa = q.transpose(1, 2)
    k_sdpa = k_full.transpose(1, 2)
    v_sdpa = v_full.transpose(1, 2)

    enable_gqa = q_sdpa.size(1) != k_sdpa.size(1)
    y_sdpa = _sdpa_attention(q_sdpa, k_sdpa, v_sdpa, window_size, enable_gqa)

    return y_sdpa.transpose(1, 2)  # back to (B, T, H, D)


# =============================================================================
# Export: flash_attn module interface (drop-in replacement for FA3)
# =============================================================================
from types import SimpleNamespace
flash_attn = SimpleNamespace(
    flash_attn_func=flash_attn_func,
    flash_attn_with_kvcache=flash_attn_with_kvcache,
)
