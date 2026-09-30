"""
Recurrent-state plumbing and the convolutional pieces: the shared causal depthwise conv, the
`short_conv` mixer, the `canon` feature, and what the cache/stats/optimizer say about a model whose
layers are not (all) attention. New flavors also run through every generic test in test_manager.py/
test_generate.py (forward/backward, save/load, optimizer groups, cached decode == naive, ragged
decode) -- but a freshly built model zero-inits every output projection, so the awake tests here
are what actually exercise the mixers.

python -m pytest modelcore/tests/test_recurrent.py -v
"""
import pytest
import torch
import torch.nn.functional as F

from modelcore import ComponentSpec, KVCache, ModelManager, OptimizerHparams
from modelcore.components.conv import ShortConv, causal_depthwise_conv
from modelcore.kernels.flash_attn import build_doc_args
from modelcore.roles import collect_param_roles
from modelcore.tests.conftest import FLAVORS, _canon, _mixed_like, _plain_like, build

CONV_FLAVORS = ["conv_only", "hybrid_attn_conv", "hybrid_win_canon", "llama_canon_mixer_only"]
BOS = 1


def _awake(model, seed=0, std=0.1):
    """Random values everywhere, so no zero-init projection hides the layer under test."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * std)
    return model


# -----------------------------------------------------------------------------
# causal_depthwise_conv

def _reference_conv(x, w):
    """F.conv1d with left padding: the textbook causal depthwise convolution."""
    B, T, D = x.shape
    K = w.size(1)
    return F.conv1d(F.pad(x.transpose(1, 2), (K - 1, 0)), w.unsqueeze(1), groups=D).transpose(1, 2)


@pytest.mark.parametrize("K", [1, 2, 4])
def test_causal_conv_matches_conv1d(K):
    torch.manual_seed(0)
    x, w = torch.randn(2, 9, 6), torch.randn(6, K)
    y, _ = causal_depthwise_conv(x, w)
    assert torch.allclose(y, _reference_conv(x, w), atol=1e-6)


@pytest.mark.parametrize("K", [2, 4])
def test_causal_conv_is_causal(K):
    torch.manual_seed(0)
    x, w = torch.randn(1, 8, 4), torch.randn(4, K)
    x2 = x.clone()
    x2[:, 5:] += 3.0
    y, y2 = causal_depthwise_conv(x, w)[0], causal_depthwise_conv(x2, w)[0]
    assert torch.equal(y[:, :5], y2[:, :5])


@pytest.mark.parametrize("K", [2, 4])
@pytest.mark.parametrize("chunks", [[9], [5, 4], [1] * 9, [2, 1, 3, 3]])
def test_chunked_with_carried_state_equals_full(K, chunks):
    """Prefill-then-decode is the same arithmetic as one full pass, whatever the split (even
    chunks shorter than K-1)."""
    torch.manual_seed(0)
    x, w = torch.randn(2, 9, 6), torch.randn(6, K)
    full, _ = causal_depthwise_conv(x, w)
    outs, state, at = [], None, 0
    for n in chunks:
        y, state = causal_depthwise_conv(x[:, at:at + n], w, state)
        outs.append(y)
        at += n
    assert state.shape == (2, K - 1, 6)
    assert torch.allclose(torch.cat(outs, dim=1), full, atol=1e-6)


def test_doc_ids_make_a_packed_row_equal_its_documents():
    torch.manual_seed(0)
    K, w = 4, torch.randn(5, 4)
    docs = [torch.randn(1, n, 5) for n in (3, 1, 6)]
    packed = torch.cat(docs, dim=1)
    doc_ids = torch.cat([torch.full((1, n), i) for i, n in enumerate((3, 1, 6))], dim=1)
    y, _ = causal_depthwise_conv(packed, w, doc_ids=doc_ids)
    expected = torch.cat([causal_depthwise_conv(d, w)[0] for d in docs], dim=1)
    assert torch.allclose(y, expected, atol=1e-6)
    leaky, _ = causal_depthwise_conv(packed, w)
    assert not torch.allclose(leaky, expected, atol=1e-3), "without doc_ids the conv must reach across documents"


# -----------------------------------------------------------------------------
# the short_conv mixer

def _mixer(n_embd=16, K=4, seed=0):
    m = ShortConv(n_embd, K)
    m.bind_layer(0)
    m.init_weights()
    _awake(m, seed)
    return m


def test_short_conv_is_gated_conv_of_b_times_v():
    torch.manual_seed(0)
    m = _mixer()
    x = torch.randn(2, 7, 16)
    b, c, v = (x @ m.in_proj.weight.T).chunk(3, dim=-1)
    expected = (c * _reference_conv(b * v, m.conv_weight)) @ m.out_proj.weight.T
    assert torch.allclose(m(x, None, None), expected, atol=1e-5)


def test_short_conv_mixer_packed_with_doc_args_equals_per_document():
    torch.manual_seed(0)
    m = _mixer()
    lens = (5, 3, 4)
    docs = [torch.randint(2, 50, (1, n)) for n in lens]
    idx = torch.cat([torch.cat([torch.tensor([[BOS]]), d[:, 1:]], dim=1) for d in docs], dim=1)
    x = torch.randn(1, idx.size(1), 16)
    with torch.no_grad():
        packed = m(x, idx, None, doc_args=build_doc_args(idx, BOS))
        parts, at = [], 0
        for n in lens:
            parts.append(m(x[:, at:at + n], idx[:, at:at + n], None))
            at += n
    assert torch.allclose(packed, torch.cat(parts, dim=1), atol=1e-5)


def test_short_conv_cached_steps_equal_full_forward():
    torch.manual_seed(0)
    m = _mixer()
    x = torch.randn(2, 8, 16)
    cache = KVCache(2, 0, 8, 0, 0, torch.device("cpu"), torch.float32)
    with torch.no_grad():
        full = m(x, None, None)
        pre = m(x[:, :5], None, cache)
        steps = [m(x[:, t:t + 1], None, cache) for t in range(5, 8)]
    assert torch.allclose(torch.cat([pre] + steps, dim=1), full, atol=1e-5)


def test_short_conv_layer_spec_and_flops():
    spec = _mixer(16, 4).layer_spec()
    assert (spec.kind, spec.state_elems, spec.fwd_flops_per_token) == ("short_conv", 3 * 16, 2 * 4 * 16 + 2 * 16)


# -----------------------------------------------------------------------------
# whole models, awake

def _cached_logits(manager, model, tokens, prefill_len):
    """Logits at every position from prefill(tokens[:prefill_len]) then one cached step per token."""
    cache = manager.new_kv_cache(model, batch_size=tokens.size(0), seq_len=tokens.size(1))
    out = [model(tokens[:, :prefill_len], kv_cache=cache)]
    for t in range(prefill_len, tokens.size(1)):
        out.append(model(tokens[:, t:t + 1], kv_cache=cache))
    return torch.cat(out, dim=1)


@pytest.mark.parametrize("flavor", CONV_FLAVORS)
@pytest.mark.parametrize("prefill_len", [1, 3, 9])
def test_cached_decode_equals_full_forward_when_awake(manager, flavor, prefill_len):
    model = _awake(build(manager, FLAVORS[flavor]()))
    tokens = torch.randint(0, 100, (2, 12), generator=torch.Generator().manual_seed(3))
    with torch.no_grad():
        full = model(tokens)
        cached = _cached_logits(manager, model, tokens, prefill_len)
    assert torch.allclose(full, cached, atol=1e-4, rtol=1e-4), (full - cached).abs().max()


@pytest.mark.parametrize("flavor", CONV_FLAVORS)
def test_awake_model_actually_uses_its_conv_layers(manager, flavor):
    """A change to an early token must reach later logits through the conv path: zero every
    attention weight and the logits still depend on the earlier tokens (conv-only models have
    nothing else to carry that)."""
    model = _awake(build(manager, FLAVORS[flavor]()))
    a = torch.randint(0, 100, (1, 10), generator=torch.Generator().manual_seed(4))
    b = a.clone()
    b[0, 6] = (b[0, 6] + 1) % 100
    with torch.no_grad():
        la, lb = model(a), model(b)
    assert torch.equal(la[:, :6], lb[:, :6])
    assert not torch.allclose(la[:, 6:], lb[:, 6:])


def _packed_batch(vocab, lens, seed=5):
    g = torch.Generator().manual_seed(seed)
    docs = []
    for n in lens:
        d = torch.randint(2, vocab, (1, n), generator=g)
        d[0, 0] = BOS
        docs.append(d)
    return docs, torch.cat(docs, dim=1)


@pytest.mark.parametrize("flavor", CONV_FLAVORS)
def test_packed_forward_with_doc_args_equals_each_document_alone(manager, flavor):
    """Intra-document masking must hold for every layer kind: conv/canon state resets at a
    document boundary just like attention's mask does."""
    config = FLAVORS[flavor]()
    model = _awake(build(manager, config))
    lens = (6, 4, 7)
    docs, packed = _packed_batch(config.vocab_size, lens)
    with torch.no_grad():
        got = model(packed, doc_args=build_doc_args(packed, BOS))
        expected = torch.cat([model(d) for d in docs], dim=1)
    assert torch.allclose(got, expected, atol=1e-4, rtol=1e-4), (got - expected).abs().max()
    with torch.no_grad():
        leaky = model(packed)
    assert not torch.allclose(leaky, expected, atol=1e-4), "without doc_args later documents must see earlier ones"


# -----------------------------------------------------------------------------
# cache, stats, roles

def test_prefill_expands_state_of_any_rank():
    src = KVCache(1, 0, 4, 0, 0, torch.device("cpu"), torch.float32)
    src.state["conv"] = torch.arange(24.0).view(1, 3, 2, 4)  # rank 4
    src.state["h"] = torch.arange(5.0).view(1, 5)             # rank 2
    dst = KVCache(3, 0, 4, 0, 0, torch.device("cpu"), torch.float32)
    dst.prefill(src)
    assert dst.state["conv"].shape == (3, 3, 2, 4) and dst.state["h"].shape == (3, 5)
    assert torch.equal(dst.state["conv"][2], src.state["conv"][0])


def test_prefill_row_carries_recurrent_state_of_any_rank():
    dst = KVCache(2, 0, 4, 0, 0, torch.device("cpu"), torch.float32)
    for row, value in enumerate((1.0, 2.0)):
        src = KVCache(1, 0, 4, 0, 0, torch.device("cpu"), torch.float32)
        src.state["s"] = torch.full((1, 3, 2, 2), value)
        src.advance(3 + row)
        dst.prefill_row(row, src)
    assert dst.state["s"][0].eq(1.0).all() and dst.state["s"][1].eq(2.0).all()
    assert dst.cache_seqlens.tolist() == [3, 4]


def test_a_model_with_no_attention_has_zero_kv_slots(manager):
    config = FLAVORS["conv_only"]()
    stats = manager.stats(config)
    assert stats.kv_cache_spec["num_kv_slots"] == 0
    assert stats.kv_bytes_per_token() == 0 and stats.kv_read_bytes(1000) == 0
    assert not stats.has_sliding_window
    assert stats.shape_summary["n_head"] is None
    model = build(manager, config)
    cache = manager.new_kv_cache(model, batch_size=2, seq_len=16)
    assert cache.n_slots == 0 and cache.k_cache.numel() == 0


def test_state_is_constant_per_row_and_counted_from_mixers_and_features(manager):
    D, K, L = 64, 4, 4
    assert manager.stats(FLAVORS["conv_only"]()).state_elems_per_row == L * (K - 1) * D
    assert manager.stats(FLAVORS["conv_only"]()).state_bytes_per_row() == L * (K - 1) * D * 4
    assert manager.stats(FLAVORS["hybrid_attn_conv"]()).state_elems_per_row == 2 * (K - 1) * D  # 2 conv layers
    # hybrid_win_canon: 2 conv mixers + canon (two sites) on all 4 blocks
    assert manager.stats(FLAVORS["hybrid_win_canon"]()).state_elems_per_row == 2 * (K - 1) * D + 4 * 2 * (K - 1) * D
    assert manager.stats(FLAVORS["gpt"]()).state_elems_per_row == 0


def test_conv_flops_are_counted_beyond_the_matmuls(manager):
    D, K = 64, 4
    stats = manager.stats(FLAVORS["conv_only"]())
    per_layer = 2 * K * D + 2 * D
    assert stats.extra_fwd_flops_per_token == 4 * per_layer
    assert stats.flops_per_token == 6 * stats.num_matmul_params + 3 * 4 * per_layer
    assert stats.decode_flops(10) == stats.decode_flops(10_000) == 2 * stats.num_matmul_params + 4 * per_layer
    assert stats.prefill_flops(7) == 7 * stats.decode_flops(1)
    canon = manager.stats(FLAVORS["llama_canon_mixer_only"]())
    assert canon.extra_fwd_flops_per_token == 3 * (2 * 3 + 1) * D  # one site, three blocks


def test_a_hybrid_prices_attention_and_conv_layers_separately(manager):
    hybrid, attn_only = manager.stats(FLAVORS["hybrid_attn_conv"]()), manager.stats(_mixed_like(["attn"] * 4))
    assert hybrid.kv_bytes_per_token() == attn_only.kv_bytes_per_token() // 2
    assert hybrid.decode_flops(100) != attn_only.decode_flops(100)


def test_conv_weights_get_their_own_adamw_role_and_group(manager):
    model = build(manager, FLAVORS["hybrid_win_canon"]())
    roles = collect_param_roles(model)
    conv_names = {n for n, p in model.named_parameters() if n.endswith("conv_weight") or ".weights." in n}
    assert conv_names
    assert {id(p) for n, p in model.named_parameters() if n in conv_names} == {id(p) for p in roles["conv"]}
    optimizer = manager.create_optimizer(model, OptimizerHparams(conv_lr=0.05))
    groups = [g for g in optimizer.param_groups if any(id(p) in {id(q) for q in roles["conv"]} for p in g["params"])]
    assert len(groups) == 1 and groups[0]["kind"] == "adamw" and groups[0]["weight_decay"] == 0.0
    assert groups[0] is optimizer.param_groups[-1], "new roles go last: the policy order is the on-disk layout"


def test_state_dict_keys(manager):
    keys = set(build(manager, FLAVORS["hybrid_win_canon"]()).state_dict())
    assert "body.blocks.0.mixer.conv_weight" in keys and "body.blocks.0.mixer.in_proj.weight" in keys
    assert "body.blocks.0.features.canon.weights.pre_mixer" in keys
    assert "body.blocks.1.features.canon.weights.pre_ffn" in keys


# -----------------------------------------------------------------------------
# validation

def _errors(manager, config):
    return [f"{e.path}: {e.message}" for e in manager.validate_config(config).errors]


def test_short_conv_kernel_size_is_validated(manager):
    config = _mixed_like(["conv", "conv"])
    config.body.params["blocks"][0].params["mixer"].params["kernel_size"] = 0
    assert any("kernel_size" in e for e in _errors(manager, config))


@pytest.mark.parametrize("sites,ok", [(["pre_mixer"], True), (["pre_ffn", "pre_mixer"], True),
                                       ([], False), (["pre_mixer", "pre_mixer"], False), (["mixer"], False)])
def test_canon_sites_are_validated(manager, sites, ok):
    config = _mixed_like(["attn"] * 2, canon=_canon(sites=tuple(sites)))
    assert manager.validate_config(config).ok == ok


def test_canon_needs_a_host_with_its_hooks(manager):
    config = _plain_like()
    config.body.params["features"] = [_canon()()]  # the stack has no pre_mixer/pre_ffn
    assert any("does not expose hook" in e for e in _errors(manager, config))
    config = _plain_like()
    config.body.params["blocks"][0].params["mixer"].params["features"] = [_canon()()]  # nor does attention
    assert any("does not expose hook" in e for e in _errors(manager, config))


def test_a_hybrid_with_implicit_kv_slots_is_rejected_with_a_clear_message(manager):
    """kv_slot=null means layer_idx, which is not a contiguous slot number once a layer isn't
    attention -- so a hybrid states its slots (the flavors do)."""
    config = _mixed_like(["conv", "attn"])
    config.body.params["blocks"][1].params["mixer"].params["kv_slot"] = None
    assert any("contiguous" in e for e in _errors(manager, config))


def test_a_mixed_model_round_trips_through_save_and_load(manager, tmp_path):
    from modelcore.store import FileSystemStore
    model = _awake(build(manager, FLAVORS["hybrid_win_canon"]()))
    store = FileSystemStore(str(tmp_path), 0)
    manager.save_model(model, store)
    again = manager.load_model(store, device=torch.device("cpu"))
    idx = torch.randint(0, 100, (2, 9))
    with torch.no_grad():
        assert torch.equal(model(idx), again(idx))
