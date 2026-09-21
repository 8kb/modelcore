"""
Test the reusable building blocks under modelcore/components/. CPU-only, no real model needed --
these operate directly on small tensors. (The window-pattern derivation rule this file used to
also cover lives outside modelcore -- see tests/test_architectures.py's
test_compute_window_sizes_* -- since it's a depth-dial policy, not a component.)

python -m pytest modelcore/tests/test_components.py -v
"""

import torch
import torch.nn.functional as F

from modelcore.components.rope import apply_rotary_emb, precompute_rotary_embeddings
from modelcore.components.norm import LayerNorm, RMSNorm
from modelcore.components.linear import Linear
from modelcore.components.mlp import ACTIVATIONS, MLP, GatedMLP
from modelcore.components.embedding import Smear
from modelcore.components.unembedding import LMHead
from modelcore.components.rotary import RotaryEmbedding


# -----------------------------------------------------------------------------
# rope

def test_apply_rotary_emb_preserves_norm():
    """Rotation is norm-preserving per (x1, x2) pair, so the full head vector's norm is unchanged."""
    torch.manual_seed(0)
    head_dim, seq_len = 8, 16
    cos, sin = precompute_rotary_embeddings(seq_len, head_dim, device="cpu", dtype=torch.float32)
    x = torch.randn(1, seq_len, 2, head_dim)  # (B, T, H, D)
    y = apply_rotary_emb(x, cos, sin)
    assert torch.allclose(x.norm(dim=-1), y.norm(dim=-1), atol=1e-5)


def test_apply_rotary_emb_relative_position_invariance():
    """q . k after RoPE depends only on the relative offset (j - i), not on absolute positions."""
    torch.manual_seed(0)
    head_dim, seq_len = 8, 32
    cos, sin = precompute_rotary_embeddings(seq_len, head_dim, device="cpu", dtype=torch.float32)
    q_vec = torch.randn(1, 1, 1, head_dim)
    k_vec = torch.randn(1, 1, 1, head_dim)

    def dot_at(i, j):
        q_rot = apply_rotary_emb(q_vec, cos[:, i:i + 1], sin[:, i:i + 1])
        k_rot = apply_rotary_emb(k_vec, cos[:, j:j + 1], sin[:, j:j + 1])
        return (q_rot * k_rot).sum().item()

    same_offset_a = dot_at(5, 2)    # offset 3
    same_offset_b = dot_at(20, 17)  # offset 3
    different_offset = dot_at(20, 10)  # offset 10
    assert abs(same_offset_a - same_offset_b) < 1e-4
    assert abs(same_offset_a - different_offset) > 1e-4


# -----------------------------------------------------------------------------
# norm

def test_norm_gives_unit_rms():
    x = torch.randn(4, 8, 16) * 5.0 + 3.0
    y = RMSNorm(eps=None)(x)
    rms = y.pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-4)


def test_rms_norm_eps_none_is_torchs_own_default():
    """eps=None is what every pre-configurable-norm model used (F.rms_norm's unset eps), so the
    v1->v2 converter's `"eps": null` must reproduce it bit for bit."""
    x = torch.randn(3, 5, 16)
    assert torch.equal(RMSNorm(eps=None)(x), F.rms_norm(x, (16,)))


def test_layer_norm_zero_mean_unit_var():
    x = torch.randn(4, 8, 16) * 5.0 + 3.0
    y = LayerNorm(eps=1e-5)(x)
    assert torch.allclose(y.mean(dim=-1), torch.zeros(4, 8), atol=1e-4)
    assert torch.allclose(y.var(dim=-1, unbiased=False), torch.ones(4, 8), atol=1e-3)


# -----------------------------------------------------------------------------
# Linear

def test_linear_casts_activations_but_keeps_fp32_master_weight():
    lin = Linear(4, 8, bias=False)
    assert lin.weight.dtype == torch.float32
    x64 = torch.randn(2, 4, dtype=torch.float64)
    y = lin(x64)
    assert y.dtype == torch.float64  # matmul ran in the activation dtype
    assert lin.weight.dtype == torch.float32  # master weight untouched


# -----------------------------------------------------------------------------
# MLP variants

def test_gated_mlp_shape_and_finite():
    mlp = GatedMLP(n_embd=32, activation="silu", hidden_dim=96)
    mlp.init_weights()
    x = torch.randn(2, 5, 32)
    y = mlp(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_mlp_inner_width_is_whatever_the_config_says():
    """hidden_dim is free -- not 4 * n_embd, and not rounded to anything."""
    for cls in (MLP, GatedMLP):
        mlp = cls(n_embd=32, activation="silu", hidden_dim=50)
        assert mlp.hidden_dim == 50
        first = mlp.c_fc if cls is MLP else mlp.gate_proj
        last = mlp.c_proj if cls is MLP else mlp.down_proj
        assert first.weight.shape == (50, 32) and last.weight.shape == (32, 50)


def test_mlp_activation_is_selectable():
    """Same weights, different activation -> different output; and relu2 is exactly relu(x)^2."""
    torch.manual_seed(0)
    x = torch.randn(2, 5, 32)
    outs = {}
    for name in ACTIVATIONS:
        mlp = MLP(n_embd=32, activation=name, hidden_dim=64)
        torch.manual_seed(1)
        torch.nn.init.uniform_(mlp.c_fc.weight, -0.3, 0.3)
        torch.nn.init.uniform_(mlp.c_proj.weight, -0.3, 0.3)  # nonzero, so the activation is visible
        outs[name] = mlp(x)
    assert len({o.flatten()[0].item() for o in outs.values()}) == len(ACTIVATIONS)
    z = torch.randn(7)
    assert torch.equal(ACTIVATIONS["relu2"](z), F.relu(z).square())


# -----------------------------------------------------------------------------
# Smear (modelcore.components.embedding)

def test_smear_token_by_token_decode_matches_full_sequence():
    """Feeding one token at a time through the KV-cache decode path must reproduce, position by
    position, the same result as running the full sequence through the training path at once."""
    torch.manual_seed(0)
    n_embd, T = 8, 5
    smear = Smear(gate_channels=4)
    smear.init_weights()
    x = torch.randn(1, T, n_embd)

    full = smear(x, kv_cache=None)

    class FakeCache:
        def __init__(self):
            self.state = {}

    cache = FakeCache()
    decoded = torch.cat([smear(x[:, t:t + 1], kv_cache=cache) for t in range(T)], dim=1)
    assert torch.allclose(full, decoded, atol=1e-6)


def test_smear_prefill_matches_full_sequence_and_caches_last_position():
    """The kv_cache-present, T>1 (prefill) branch computes the identical formula as the
    kv_cache=None (training) branch, plus writing kv_cache.state for the next decode step."""
    torch.manual_seed(0)
    n_embd, T = 8, 5
    smear = Smear(gate_channels=4)
    smear.init_weights()
    x = torch.randn(1, T, n_embd)

    full = smear(x, kv_cache=None)

    class FakeCache:
        def __init__(self):
            self.state = {}

    cache = FakeCache()
    prefill = smear(x, kv_cache=cache)
    assert torch.allclose(full, prefill, atol=1e-6)
    assert torch.equal(cache.state["prev_embedding"], x[:, -1:, :])


# -----------------------------------------------------------------------------
# LMHead (modelcore.components.unembedding)

def test_lm_head_softcap_bounds_logits_and_crops_vocab():
    n_embd, vocab_size, padded = 8, 20, 32
    head = LMHead(n_embd, vocab_size, padded, RMSNorm(eps=None), softcap=15)
    torch.manual_seed(0)
    torch.nn.init.normal_(head.lm_head.weight, mean=0.0, std=100.0)  # force large pre-softcap logits
    x = torch.randn(2, 3, n_embd) * 50
    logits = head(x)
    assert logits.shape == (2, 3, vocab_size)
    assert torch.all(logits.abs() <= 15.0 + 1e-4)


def test_lm_head_loss_path_matches_manual_cross_entropy():
    n_embd, vocab_size, padded = 8, 20, 32
    head = LMHead(n_embd, vocab_size, padded, RMSNorm(eps=None), softcap=15)
    head.init_weights()
    x = torch.randn(2, 3, n_embd)
    targets = torch.randint(0, vocab_size, (2, 3))
    loss = head(x, targets=targets)
    logits = head(x)
    manual = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1), ignore_index=-1)
    assert torch.allclose(loss, manual, atol=1e-5)


def test_lm_head_tied_weight_shares_storage_and_declares_no_role():
    n_embd, vocab_size, padded = 8, 20, 32
    shared = torch.nn.Parameter(torch.randn(padded, n_embd))
    head = LMHead(n_embd, vocab_size, padded, RMSNorm(eps=None), softcap=15, weight=shared)
    assert head.lm_head.weight is shared
    assert head.param_roles() == {}


# -----------------------------------------------------------------------------
# RotaryEmbedding (modelcore.components.rotary)

def test_rotary_embedding_offset_matches_manual_slice():
    head_dim, seq_len = 8, 16
    rope = RotaryEmbedding(head_dim, seq_len, over_compute=10)
    rope.init_weights()
    T0, T = 5, 4
    q = torch.randn(1, T, 2, head_dim)
    k = torch.randn(1, T, 2, head_dim)

    class FakeCache:
        def get_pos(self):
            return T0

    q_rot, k_rot = rope(q, k, FakeCache())
    cos_manual, sin_manual = rope.cos[:, T0:T0 + T], rope.sin[:, T0:T0 + T]
    assert torch.equal(q_rot, apply_rotary_emb(q, cos_manual, sin_manual))
    assert torch.equal(k_rot, apply_rotary_emb(k, cos_manual, sin_manual))


def test_rotary_embedding_no_cache_uses_zero_offset():
    head_dim, seq_len = 8, 16
    rope = RotaryEmbedding(head_dim, seq_len, over_compute=10)
    rope.init_weights()
    q = torch.randn(1, 3, 2, head_dim)
    k = torch.randn(1, 3, 2, head_dim)
    q_rot, k_rot = rope(q, k, kv_cache=None)
    assert torch.equal(q_rot, apply_rotary_emb(q, rope.cos[:, :3], rope.sin[:, :3]))
    assert torch.equal(k_rot, apply_rotary_emb(k, rope.cos[:, :3], rope.sin[:, :3]))
