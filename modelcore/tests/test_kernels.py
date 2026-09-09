"""
Test Flash Attention unified interface - verify FA3 and SDPA produce identical results.

Run: python -m pytest modelcore/tests/test_kernels.py -v -s

Note on test structure:
    Tests are split into two classes due to dtype/device constraints:

    1. TestFA3VsSDPA: Comparison tests that run both FA3 and SDPA on the same inputs
       and verify they produce identical results. These require a compatible GPU (FA3 only
       works on sm80 and sm90) and use bfloat16 (FA3 doesn't support float32).

    2. TestSDPAOnly: Tests that only exercise the SDPA fallback path. These can run
       on any device (CUDA, CPU, MPS) with the appropriate dtype for that device.
"""
import torch
import torch.nn.functional as F
import pytest
import modelcore.kernels.flash_attn as fa_module
from modelcore.kernels.flash_attn import flash_attn, build_doc_args, HAS_FA3
from modelcore.cache import KVCache


def set_impl(impl):
    """Set the implementation override ('fa3', 'sdpa', or None for auto) and re-resolve USE_FA3."""
    fa_module._override_impl = impl
    fa_module.USE_FA3 = fa_module._resolve_use_fa3()


def run_both_impls(fn):
    """Run a function with both FA3 and SDPA, return both outputs."""
    set_impl('fa3')
    out_fa3 = fn()
    set_impl('sdpa')
    out_sdpa = fn()
    set_impl(None)  # reset
    return out_fa3, out_sdpa


def assert_close(t1, t2, name, atol=1e-2, rtol=1e-2):
    """Assert two tensors are close, with helpful error message."""
    max_diff = (t1 - t2).abs().max().item()
    mean_diff = (t1 - t2).abs().mean().item()
    assert torch.allclose(t1, t2, atol=atol, rtol=rtol), \
        f"{name}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}"
    return max_diff, mean_diff


# =============================================================================
# FA3 load-failure diagnostics
# =============================================================================
def test_fa3_load_error_reported_when_no_cuda(monkeypatch):
    """_load_flash_attention_3() must return a reason string, not swallow it, when there's no
    CUDA device -- this is the real path exercised on every CPU/MPS dev machine, and used to be
    indistinguishable from any other failure (a bare `except Exception: return None`)."""
    monkeypatch.setattr(fa_module.torch.cuda, "is_available", lambda: False)
    module, reason = fa_module._load_flash_attention_3()
    assert module is None
    assert reason == "no CUDA device available"


def test_fa3_load_error_captures_exception_text(monkeypatch):
    """A real exception during loading (HF hub unreachable, kernels import broken, etc.) must be
    captured into the reason string, not discarded -- this is the actual bug found running the
    Stage 5 architecture contest on a real 4x A100 pod: FA3 fell back to SDPA with zero indication
    of why, even though the GPU should have been supported."""
    monkeypatch.setattr(fa_module.torch.cuda, "is_available", lambda: True)

    def _boom():
        raise RuntimeError("simulated HF hub failure")
    monkeypatch.setattr(fa_module.torch.cuda, "get_device_capability", _boom)

    module, reason = fa_module._load_flash_attention_3()
    assert module is None
    assert reason == "RuntimeError: simulated HF hub failure"


# =============================================================================
# FA3 vs SDPA comparison tests
# =============================================================================
@pytest.mark.skipif(not HAS_FA3, reason="FA3 required to compare implementations")
class TestFA3VsSDPA:
    """Compare FA3 and SDPA produce identical results."""

    DEVICE = "cuda"
    DTYPE = torch.bfloat16

    def test_basic_causal(self):
        """Basic causal attention."""
        B, T, H, D = 2, 64, 4, 32
        q = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        k = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        v = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)

        def run():
            return flash_attn.flash_attn_func(q, k, v, causal=True, window_size=(T, 0))

        y_fa3, y_sdpa = run_both_impls(run)
        max_diff, mean_diff = assert_close(y_fa3, y_sdpa, "basic_causal")
        print(f"basic_causal: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

    def test_full_context(self):
        """Full context (window_size=-1)."""
        B, T, H, D = 2, 128, 4, 32
        q = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        k = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        v = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)

        def run():
            return flash_attn.flash_attn_func(q, k, v, causal=True, window_size=(-1, -1))

        y_fa3, y_sdpa = run_both_impls(run)
        max_diff, mean_diff = assert_close(y_fa3, y_sdpa, "full_context")
        print(f"full_context: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

    def test_sliding_window(self):
        """Sliding window attention."""
        B, T, H, D = 2, 128, 4, 32
        window = 32
        q = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        k = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        v = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)

        def run():
            return flash_attn.flash_attn_func(q, k, v, causal=True, window_size=(window, 0))

        y_fa3, y_sdpa = run_both_impls(run)
        max_diff, mean_diff = assert_close(y_fa3, y_sdpa, "sliding_window")
        print(f"sliding_window: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

    def test_gqa(self):
        """Group Query Attention (fewer KV heads than Q heads)."""
        B, T, D = 2, 64, 32
        n_heads = 8
        n_kv_heads = 2

        q = torch.randn(B, T, n_heads, D, device=self.DEVICE, dtype=self.DTYPE)
        k = torch.randn(B, T, n_kv_heads, D, device=self.DEVICE, dtype=self.DTYPE)
        v = torch.randn(B, T, n_kv_heads, D, device=self.DEVICE, dtype=self.DTYPE)

        def run():
            return flash_attn.flash_attn_func(q, k, v, causal=True, window_size=(T, 0))

        y_fa3, y_sdpa = run_both_impls(run)
        max_diff, mean_diff = assert_close(y_fa3, y_sdpa, "gqa")
        print(f"gqa: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

    def test_larger_model(self):
        """Larger dimensions closer to real model."""
        B, T, H, D = 4, 256, 12, 64
        q = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        k = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        v = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)

        def run():
            return flash_attn.flash_attn_func(q, k, v, causal=True, window_size=(-1, -1))

        y_fa3, y_sdpa = run_both_impls(run)
        max_diff, mean_diff = assert_close(y_fa3, y_sdpa, "larger_model")
        print(f"larger_model: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

    def test_kvcache_prefill(self):
        """Test prefill (inserting multiple tokens into empty cache)."""
        B, T_max, H, D = 2, 64, 4, 32
        T_prefill = 16

        q = torch.randn(B, T_prefill, H, D, device=self.DEVICE, dtype=self.DTYPE)
        k = torch.randn(B, T_prefill, H, D, device=self.DEVICE, dtype=self.DTYPE)
        v = torch.randn(B, T_prefill, H, D, device=self.DEVICE, dtype=self.DTYPE)

        def run():
            k_cache = torch.zeros(B, T_max, H, D, device=self.DEVICE, dtype=self.DTYPE)
            v_cache = torch.zeros(B, T_max, H, D, device=self.DEVICE, dtype=self.DTYPE)
            cache_seqlens = torch.zeros(B, dtype=torch.int32, device=self.DEVICE)
            return flash_attn.flash_attn_with_kvcache(
                q, k_cache, v_cache, k=k, v=v,
                cache_seqlens=cache_seqlens,
                causal=True, window_size=(T_max, 0)
            )

        y_fa3, y_sdpa = run_both_impls(run)
        max_diff, mean_diff = assert_close(y_fa3, y_sdpa, "prefill")
        print(f"prefill: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

    def test_kvcache_single_token(self):
        """Test single token generation (cache already has content)."""
        B, T_max, H, D = 2, 64, 4, 32
        T_prefill = 16

        k_init = torch.randn(B, T_prefill, H, D, device=self.DEVICE, dtype=self.DTYPE)
        v_init = torch.randn(B, T_prefill, H, D, device=self.DEVICE, dtype=self.DTYPE)
        q_single = torch.randn(B, 1, H, D, device=self.DEVICE, dtype=self.DTYPE)
        k_single = torch.randn(B, 1, H, D, device=self.DEVICE, dtype=self.DTYPE)
        v_single = torch.randn(B, 1, H, D, device=self.DEVICE, dtype=self.DTYPE)

        def run():
            k_cache = torch.zeros(B, T_max, H, D, device=self.DEVICE, dtype=self.DTYPE)
            v_cache = torch.zeros(B, T_max, H, D, device=self.DEVICE, dtype=self.DTYPE)
            k_cache[:, :T_prefill, :, :] = k_init
            v_cache[:, :T_prefill, :, :] = v_init
            cache_seqlens = torch.full((B,), T_prefill, dtype=torch.int32, device=self.DEVICE)
            return flash_attn.flash_attn_with_kvcache(
                q_single, k_cache, v_cache, k=k_single, v=v_single,
                cache_seqlens=cache_seqlens,
                causal=True, window_size=(T_max, 0)
            )

        y_fa3, y_sdpa = run_both_impls(run)
        max_diff, mean_diff = assert_close(y_fa3, y_sdpa, "single_token")
        print(f"single_token: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

    def test_kvcache_single_token_sliding_window(self):
        """Test single token decode with sliding window smaller than cache size.

        This catches the bug where SDPA ignores window_size during Tq=1 decode.
        When window < Tk, FA3 only attends to the last (window+1) tokens,
        but SDPA was attending to all cached tokens.
        """
        B, T_max, H, D = 2, 64, 4, 32
        T_prefill = 32  # Enough tokens to exceed window
        window = 8      # Window SMALLER than cache size

        k_init = torch.randn(B, T_prefill, H, D, device=self.DEVICE, dtype=self.DTYPE)
        v_init = torch.randn(B, T_prefill, H, D, device=self.DEVICE, dtype=self.DTYPE)
        q_single = torch.randn(B, 1, H, D, device=self.DEVICE, dtype=self.DTYPE)
        k_single = torch.randn(B, 1, H, D, device=self.DEVICE, dtype=self.DTYPE)
        v_single = torch.randn(B, 1, H, D, device=self.DEVICE, dtype=self.DTYPE)

        def run():
            k_cache = torch.zeros(B, T_max, H, D, device=self.DEVICE, dtype=self.DTYPE)
            v_cache = torch.zeros(B, T_max, H, D, device=self.DEVICE, dtype=self.DTYPE)
            k_cache[:, :T_prefill, :, :] = k_init
            v_cache[:, :T_prefill, :, :] = v_init
            cache_seqlens = torch.full((B,), T_prefill, dtype=torch.int32, device=self.DEVICE)
            return flash_attn.flash_attn_with_kvcache(
                q_single, k_cache, v_cache, k=k_single, v=v_single,
                cache_seqlens=cache_seqlens,
                causal=True, window_size=(window, 0)  # window=8 < Tk=33
            )

        y_fa3, y_sdpa = run_both_impls(run)
        max_diff, mean_diff = assert_close(y_fa3, y_sdpa, "single_token_sliding_window")
        print(f"single_token_sliding_window: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

    def test_backward_gradients_match(self):
        """Verify gradients are similar between FA3 and SDPA."""
        B, T, H, D = 2, 32, 4, 16

        q_data = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        k_data = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        v_data = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)

        def run():
            q = q_data.clone().requires_grad_(True)
            k = k_data.clone().requires_grad_(True)
            v = v_data.clone().requires_grad_(True)
            y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=(T, 0))
            loss = y.sum()
            loss.backward()
            return y.detach(), q.grad.detach(), k.grad.detach(), v.grad.detach()

        set_impl('fa3')
        y_fa3, q_grad_fa3, k_grad_fa3, v_grad_fa3 = run()
        set_impl('sdpa')
        y_sdpa, q_grad_sdpa, k_grad_sdpa, v_grad_sdpa = run()
        set_impl(None)

        max_diff, mean_diff = assert_close(y_fa3, y_sdpa, "backward_output")
        print(f"backward_output: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

        max_diff, mean_diff = assert_close(q_grad_fa3, q_grad_sdpa, "q_grad", atol=0.05, rtol=0.05)
        print(f"q_grad: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

        max_diff, mean_diff = assert_close(k_grad_fa3, k_grad_sdpa, "k_grad", atol=0.05, rtol=0.05)
        print(f"k_grad: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

        max_diff, mean_diff = assert_close(v_grad_fa3, v_grad_sdpa, "v_grad", atol=0.05, rtol=0.05)
        print(f"v_grad: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

    def test_doc_masking(self):
        """FA3 varlen vs. the SDPA doc-mask fallback, on a batch with several documents per row.
        doc_args is rebuilt per backend (rather than reused across run_both_impls) because
        build_doc_args's cu_seqlens/max_seqlen depend on the module-level USE_FA3 flag at the time
        it's called."""
        B, T, H, D = 2, 64, 4, 32
        bos = 999
        idx = torch.zeros(B, T, dtype=torch.long, device=self.DEVICE)
        idx[:, 0] = bos
        idx[:, 20] = bos
        idx[:, 45] = bos
        q = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        k = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        v = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)

        set_impl('fa3')
        doc_args_fa3 = build_doc_args(idx, bos)
        y_fa3 = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=(T, 0), doc_args=doc_args_fa3)
        set_impl('sdpa')
        doc_args_sdpa = build_doc_args(idx, bos)
        y_sdpa = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=(T, 0), doc_args=doc_args_sdpa)
        set_impl(None)

        max_diff, mean_diff = assert_close(y_fa3, y_sdpa, "doc_masking")
        print(f"doc_masking: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

    def test_doc_masking_with_window(self):
        """Same, with a sliding window narrower than the documents."""
        B, T, H, D = 2, 64, 4, 32
        window = 12
        bos = 999
        idx = torch.zeros(B, T, dtype=torch.long, device=self.DEVICE)
        idx[:, 0] = bos
        idx[:, 20] = bos
        idx[:, 45] = bos
        q = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        k = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        v = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)

        set_impl('fa3')
        doc_args_fa3 = build_doc_args(idx, bos)
        y_fa3 = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=(window, 0), doc_args=doc_args_fa3)
        set_impl('sdpa')
        doc_args_sdpa = build_doc_args(idx, bos)
        y_sdpa = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=(window, 0), doc_args=doc_args_sdpa)
        set_impl(None)

        max_diff, mean_diff = assert_close(y_fa3, y_sdpa, "doc_masking_with_window")
        print(f"doc_masking_with_window: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")


# =============================================================================
# SDPA-only tests (run on any device)
# =============================================================================
class TestSDPAOnly:
    """Test SDPA fallback works correctly. Runs on any device."""

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    def test_basic_forward(self):
        """Test SDPA forward pass produces valid output."""
        set_impl('sdpa')
        B, T, H, D = 2, 64, 4, 32
        q = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        k = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        v = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)

        y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=(T, 0))

        assert y.shape == (B, T, H, D)
        assert not torch.isnan(y).any(), "Output contains NaN"
        set_impl(None)

    def test_backward(self):
        """Test gradients flow through SDPA."""
        set_impl('sdpa')
        B, T, H, D = 2, 32, 4, 16
        q = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE, requires_grad=True)
        k = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE, requires_grad=True)
        v = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE, requires_grad=True)

        y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=(T, 0))
        loss = y.sum()
        loss.backward()

        assert q.grad is not None, "No gradient for q"
        assert k.grad is not None, "No gradient for k"
        assert v.grad is not None, "No gradient for v"
        assert not torch.isnan(q.grad).any(), "NaN in q gradient"
        set_impl(None)

    def test_kvcache(self):
        """Test SDPA with KV cache."""
        set_impl('sdpa')
        B, T_max, H, D = 2, 64, 4, 32
        n_kv_slots = 1

        cache = KVCache(
            batch_size=B, num_heads=H, seq_len=T_max, head_dim=D,
            num_kv_slots=n_kv_slots, device=self.DEVICE, dtype=self.DTYPE
        )
        k_cache, v_cache = cache.get_slot_cache(0)

        # Prefill
        T_prefill = 16
        q = torch.randn(B, T_prefill, H, D, device=self.DEVICE, dtype=self.DTYPE)
        k = torch.randn(B, T_prefill, H, D, device=self.DEVICE, dtype=self.DTYPE)
        v = torch.randn(B, T_prefill, H, D, device=self.DEVICE, dtype=self.DTYPE)

        y = flash_attn.flash_attn_with_kvcache(
            q, k_cache, v_cache, k=k, v=v,
            cache_seqlens=cache.cache_seqlens,
            causal=True, window_size=(T_max, 0)
        )
        cache.advance(T_prefill)

        assert y.shape == (B, T_prefill, H, D)
        assert cache.get_pos() == T_prefill

        # Generate single token
        q_single = torch.randn(B, 1, H, D, device=self.DEVICE, dtype=self.DTYPE)
        k_single = torch.randn(B, 1, H, D, device=self.DEVICE, dtype=self.DTYPE)
        v_single = torch.randn(B, 1, H, D, device=self.DEVICE, dtype=self.DTYPE)

        y_single = flash_attn.flash_attn_with_kvcache(
            q_single, k_cache, v_cache, k=k_single, v=v_single,
            cache_seqlens=cache.cache_seqlens,
            causal=True, window_size=(T_max, 0)
        )
        cache.advance(1)

        assert y_single.shape == (B, 1, H, D)
        assert cache.get_pos() == T_prefill + 1
        set_impl(None)


# =============================================================================
# build_doc_args: BOS-boundary segmentation (device/backend independent)
# =============================================================================
BOS = 999


class TestBuildDocArgs:
    """build_doc_args's segmentation logic is pure tensor arithmetic over idx -- no attention
    kernel involved, so these run identically everywhere."""

    def test_multi_doc_row(self):
        idx = torch.tensor([[BOS, 1, 2, BOS, 3, 4, 5]])
        doc_ids = build_doc_args(idx, BOS).doc_ids
        assert doc_ids.tolist() == [[0, 0, 0, 1, 1, 1, 1]]

    def test_trailing_bos_run_folds_into_previous_document_by_default(self):
        """padding_id defaults to None: a document made ENTIRELY of BOS (BestFitPadPacker's pad
        tail, written with bos_token_id as filler since it has no distinct padding_id) is folded
        into the preceding document rather than counted as its own -- a real document always has
        non-BOS content after its own leading BOS, so an all-BOS document can only be padding."""
        idx = torch.tensor([[BOS, 1, 2, BOS, BOS, BOS]])
        doc_ids = build_doc_args(idx, BOS).doc_ids
        assert doc_ids.tolist() == [[0, 0, 0, 0, 0, 0]]

    def test_bos_run_with_trailing_content_is_not_folded(self):
        """The fold only applies to a document with NO non-BOS content at all. A BOS run followed
        by more real content (not at the row's tail) is a real document boundary, not padding."""
        idx = torch.tensor([[BOS, 1, 2, BOS, BOS, 3, 4]])
        doc_ids = build_doc_args(idx, BOS).doc_ids
        assert doc_ids.tolist() == [[0, 0, 0, 1, 1, 1, 1]]

    def test_all_padding_row_is_not_folded(self):
        """A row that is 100% padding (source ran dry with no real content at all) has nothing to
        fold into -- left as a single degenerate document rather than going negative."""
        idx = torch.tensor([[BOS, BOS, BOS, BOS]])
        doc_ids = build_doc_args(idx, BOS).doc_ids
        assert doc_ids.tolist() == [[0, 0, 0, 0]]

    def test_explicit_padding_id_needs_no_folding(self):
        """A packer using a distinct padding_id (not bos_token_id) never produces an all-BOS
        pad tail in the first place -- the padded positions simply never trigger is_start, so they
        already inherit the preceding document's id from the cumsum with no special-casing."""
        PAD = 12345
        idx = torch.tensor([[BOS, 1, 2, PAD, PAD, PAD]])
        doc_ids = build_doc_args(idx, BOS, padding_id=PAD).doc_ids
        assert doc_ids.tolist() == [[0, 0, 0, 0, 0, 0]]

    def test_explicit_padding_id_does_not_fold_a_real_trailing_bos_document(self):
        """With padding_id given, the None-only fallback heuristic must not run at all -- a
        genuine short final document (not padding) stays its own document, not folded away."""
        PAD = 12345
        idx = torch.tensor([[BOS, 1, 2, BOS, PAD, PAD]])  # second doc is real (len 1: token BOS) + pad tail
        doc_ids = build_doc_args(idx, BOS, padding_id=PAD).doc_ids
        assert doc_ids.tolist() == [[0, 0, 0, 1, 1, 1]]

    def test_explicit_padding_id_matches_the_default_bos_fold_heuristic(self):
        """padding_id is a robustness/clarity improvement, not a correctness fix: the default
        (None -> bos-run fold-in) heuristic already produces the same document boundaries as
        passing the real padding_id explicitly would, for the row shape it's designed to handle."""
        idx_bos_padded = torch.tensor([[BOS, 1, 2, BOS, BOS, BOS]])
        heuristic_doc_ids = build_doc_args(idx_bos_padded, BOS).doc_ids

        PAD = 12345
        idx_distinct_padded = torch.tensor([[BOS, 1, 2, PAD, PAD, PAD]])
        explicit_doc_ids = build_doc_args(idx_distinct_padded, BOS, padding_id=PAD).doc_ids

        assert heuristic_doc_ids.tolist() == explicit_doc_ids.tolist() == [[0, 0, 0, 0, 0, 0]]

    def test_no_bos_is_one_document(self):
        idx = torch.tensor([[1, 2, 3, 4]])
        doc_ids = build_doc_args(idx, BOS).doc_ids
        assert doc_ids.tolist() == [[0, 0, 0, 0]]

    def test_rows_are_independent(self):
        idx = torch.tensor([
            [BOS, 1, BOS, 2],
            [BOS, 3, 4, 5],
        ])
        doc_ids = build_doc_args(idx, BOS).doc_ids
        assert doc_ids.tolist() == [[0, 0, 1, 1], [0, 0, 0, 0]]

    def test_none_when_fa3_inactive(self):
        set_impl('sdpa')
        idx = torch.tensor([[BOS, 1, 2, BOS, 3]])
        args = build_doc_args(idx, BOS)
        set_impl(None)
        assert args.cu_seqlens is None
        assert args.max_seqlen is None

    def test_cu_seqlens_when_fa3_active(self, monkeypatch):
        """cu_seqlens/max_seqlen are only populated for the varlen (FA3) path -- force the flag
        directly rather than via set_impl, since set_impl('fa3') requires a real FA3 kernel."""
        monkeypatch.setattr(fa_module, "USE_FA3", True)
        B, T = 2, 6
        idx = torch.tensor([
            [BOS, 1, 2, BOS, 3, 4],   # doc starts at row-local 0, 3 -> flat 0, 3
            [BOS, 5, BOS, BOS, 6, 7], # doc starts at row-local 0, 2 (BOS,BOS run collapses) -> flat 6, 8
        ])
        args = build_doc_args(idx, BOS)
        default_cap = fa_module.DEFAULT_MAX_DOCS_PER_ROW * B
        assert args.max_seqlen == T
        assert args.cu_seqlens.dtype == torch.int32
        assert args.cu_seqlens.shape == (default_cap + 1,)  # default cap: DEFAULT_MAX_DOCS_PER_ROW
                                                              # * B, NOT the B*T worst case -- see
                                                              # build_doc_args's docstring for why
                                                              # that worst case OOM'd a real H100 run
        num_docs = 4  # row0: [0,3]; row1: [6,8] (9 collapses into 8's run)
        assert args.cu_seqlens[:num_docs].tolist() == [0, 3, 6, 8]
        assert args.cu_seqlens[num_docs].item() == B * T
        assert (args.cu_seqlens[num_docs:] == B * T).all()  # zero-length trailing segments

    def test_default_cap_is_not_worst_case(self):
        """Regression test for the bug that OOM'd a real H100 run: the default cu_seqlens cap
        must scale with a small per-row document budget, not with B*T (every token its own
        document) -- that worst case made the FA3 varlen kernel allocate backward-pass scratch for
        a declared batch of 131,072 sequences when the real batch had ~270 documents."""
        B, T = 8, 2048
        default_cap = fa_module.DEFAULT_MAX_DOCS_PER_ROW * B
        assert default_cap < B * T
        assert default_cap == 512

    def test_max_docs_overflow_raises(self, monkeypatch):
        monkeypatch.setattr(fa_module, "USE_FA3", True)
        idx = torch.tensor([[BOS, 1, BOS, 2, BOS, 3]])  # 3 documents
        with pytest.raises(AssertionError):
            build_doc_args(idx, BOS, max_docs=2)


# =============================================================================
# Intra-document masking correctness (SDPA fallback vs. a naive per-document reference)
# =============================================================================
class TestDocMaskingSDPA:
    """Verify the SDPA doc-masked path against a reference that runs each document through
    ordinary attention separately -- the ground truth intra-document masking is supposed to match."""

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    def _naive_reference(self, q, k, v, doc_ids, window):
        """q, k, v: (B, T, H, D). Splits each row into its contiguous documents (doc_ids is
        monotonically non-decreasing within a row by construction) and runs plain causal
        (optionally windowed) attention within each, independently."""
        B, T, H, D = q.shape
        out = torch.zeros_like(q)
        for b in range(B):
            ids = doc_ids[b]
            for d in ids.unique().tolist():
                positions = (ids == d).nonzero(as_tuple=True)[0]
                start, end = positions[0].item(), positions[-1].item() + 1
                qi = q[b:b+1, start:end].transpose(1, 2)  # (1, H, t, D)
                ki = k[b:b+1, start:end].transpose(1, 2)
                vi = v[b:b+1, start:end].transpose(1, 2)
                enable_gqa = qi.size(1) != ki.size(1)
                t = qi.size(2)
                if window is not None and window >= 0 and window < t:
                    row_idx = torch.arange(t, device=qi.device).unsqueeze(1)
                    col_idx = torch.arange(t, device=qi.device).unsqueeze(0)
                    mask = (col_idx <= row_idx) & ((row_idx - col_idx) <= window)
                    yi = F.scaled_dot_product_attention(qi, ki, vi, attn_mask=mask, enable_gqa=enable_gqa)
                else:
                    yi = F.scaled_dot_product_attention(qi, ki, vi, is_causal=True, enable_gqa=enable_gqa)
                out[b:b+1, start:end] = yi.transpose(1, 2)
        return out

    def _row_with_docs(self, B, T, boundaries):
        idx = torch.zeros(B, T, dtype=torch.long, device=self.DEVICE)
        for pos in boundaries:
            idx[:, pos] = BOS
        return idx

    def test_matches_naive_reference_full_context(self):
        set_impl('sdpa')
        B, T, H, D = 2, 24, 2, 16
        idx = self._row_with_docs(B, T, [0, 7, 15])
        q = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        k = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        v = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        doc_args = build_doc_args(idx, BOS)

        y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=(-1, -1), doc_args=doc_args)
        y_ref = self._naive_reference(q, k, v, doc_args.doc_ids, window=None)
        set_impl(None)
        assert_close(y, y_ref, "doc_masking_full_context")

    def test_matches_naive_reference_with_window(self):
        set_impl('sdpa')
        B, T, H, D = 2, 24, 2, 16
        window = 4
        idx = self._row_with_docs(B, T, [0, 7, 15])
        q = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        k = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        v = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        doc_args = build_doc_args(idx, BOS)

        y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=(window, 0), doc_args=doc_args)
        y_ref = self._naive_reference(q, k, v, doc_args.doc_ids, window=window)
        set_impl(None)
        assert_close(y, y_ref, "doc_masking_with_window")

    def test_matches_naive_reference_with_gqa(self):
        set_impl('sdpa')
        B, T, D = 2, 24, 16
        n_heads, n_kv_heads = 4, 2
        idx = self._row_with_docs(B, T, [0, 10])
        q = torch.randn(B, T, n_heads, D, device=self.DEVICE, dtype=self.DTYPE)
        k = torch.randn(B, T, n_kv_heads, D, device=self.DEVICE, dtype=self.DTYPE)
        v = torch.randn(B, T, n_kv_heads, D, device=self.DEVICE, dtype=self.DTYPE)
        doc_args = build_doc_args(idx, BOS)

        y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=(-1, -1), doc_args=doc_args)
        y_ref = self._naive_reference(q, k, v, doc_args.doc_ids, window=None)
        set_impl(None)
        assert_close(y, y_ref, "doc_masking_gqa")

    def test_none_doc_args_is_bit_identical_to_current_behavior(self):
        """doc_args=None must be exactly today's code path -- the main guard on the signature
        churn this feature added everywhere doc_args was threaded through."""
        set_impl('sdpa')
        B, T, H, D = 2, 16, 2, 8
        q = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        k = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        v = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)

        y_implicit = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=(T, 0))
        y_explicit_none = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=(T, 0), doc_args=None)
        set_impl(None)
        assert torch.equal(y_implicit, y_explicit_none)


# =============================================================================
# Override mechanism tests
# =============================================================================
class TestOverrideMechanism:
    """Test that the override mechanism works correctly."""

    @pytest.mark.skipif(not HAS_FA3, reason="FA3 required")
    def test_override_fa3(self):
        """Test that override='fa3' uses FA3."""
        set_impl('fa3')
        assert fa_module.USE_FA3 == True
        set_impl(None)

    def test_override_sdpa(self):
        """Test that override='sdpa' uses SDPA."""
        set_impl('sdpa')
        assert fa_module.USE_FA3 == False
        set_impl(None)

    def test_override_auto(self):
        """Test that override=None uses auto-detection."""
        set_impl(None)
        assert fa_module.USE_FA3 == HAS_FA3


if __name__ == "__main__":
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
        major, minor = torch.cuda.get_device_capability()
        print(f"Compute capability: {major}.{minor}")
    print(f"HAS_FA3: {HAS_FA3}")
    print()

    pytest.main([__file__, "-v", "-s"])
