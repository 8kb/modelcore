"""
The SSD scan (modelcore.kernels.ssm) and the Mamba-2 mixer. The chunked reference is checked against a
plain sequential recurrence -- the definition of the model -- and, on a GPU, mamba_ssm's kernels
are checked against the reference.

python -m pytest modelcore/tests/test_ssm.py -v
"""
import math

import pytest
import torch

import modelcore.kernels.ssm as ssm
from modelcore import KVCache, OptimizerHparams
from modelcore.components.mamba2 import Mamba2
from modelcore.kernels.ssm import ssd_scan, ssd_scan_reference, ssd_step_reference
from modelcore.roles import collect_param_roles
from modelcore.tests.conftest import FLAVORS, RMS_NORM, _mamba2_mixer, _mixed_like, build
from modelcore.kernels.flash_attn import build_doc_args


def _inputs(B=2, T=13, H=4, P=6, G=2, N=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g)
    x, Bm, Cm = r(B, T, H, P), r(B, T, G, N), r(B, T, G, N)
    dt = torch.nn.functional.softplus(r(B, T, H) - 2)
    A = -torch.exp(r(H) * 0.5)
    D = r(H)
    return x, dt, A, Bm, Cm, D


def _sequential(x, dt, A, Bm, Cm, D, state=None, doc_ids=None):
    """h_t = exp(dt_t A) h_{t-1} + dt_t x_t (x) B_t; y_t = h_t . C_t + D x_t -- the definition."""
    B, T, H, P = x.shape
    G, N = Bm.size(2), Bm.size(3)
    h = torch.zeros(B, H, P, N) if state is None else state.clone()
    ys = []
    for t in range(T):
        if doc_ids is not None and t > 0:
            new_doc = (doc_ids[:, t] != doc_ids[:, t - 1]).view(B, 1, 1, 1)
            h = h.masked_fill(new_doc, 0.0)
        Bh, Ch = Bm[:, t].repeat_interleave(H // G, dim=1), Cm[:, t].repeat_interleave(H // G, dim=1)
        h = h * torch.exp(dt[:, t] * A).view(B, H, 1, 1) + (dt[:, t, :, None] * x[:, t])[..., None] * Bh[:, :, None, :]
        ys.append(torch.einsum("bhpn,bhn->bhp", h, Ch) + D.view(1, H, 1) * x[:, t])
    return torch.stack(ys, dim=1), h


@pytest.mark.parametrize("chunk", [1, 4, 5, 8, 13, 32])
def test_chunked_reference_equals_the_sequential_recurrence(chunk):
    args = _inputs()
    y, h = ssd_scan_reference(*args, chunk)
    y0, h0 = _sequential(*args)
    assert torch.allclose(y, y0, atol=1e-4, rtol=1e-4) and torch.allclose(h, h0, atol=1e-4, rtol=1e-4)


def test_no_D_and_a_single_group():
    x, dt, A, Bm, Cm, _ = _inputs(G=1)
    y, _ = ssd_scan_reference(x, dt, A, Bm, Cm, None, 4)
    assert torch.allclose(y, _sequential(x, dt, A, Bm, Cm, torch.zeros(4))[0], atol=1e-4)


@pytest.mark.parametrize("chunk", [4, 8])
def test_an_initial_state_is_carried_and_the_final_state_resumes(chunk):
    x, dt, A, Bm, Cm, D = _inputs(T=12)
    full, hfull = ssd_scan_reference(x, dt, A, Bm, Cm, D, chunk)
    cut = 5
    y1, h1 = ssd_scan_reference(x[:, :cut], dt[:, :cut], A, Bm[:, :cut], Cm[:, :cut], D, chunk)
    y2, h2 = ssd_scan_reference(x[:, cut:], dt[:, cut:], A, Bm[:, cut:], Cm[:, cut:], D, chunk, initial_state=h1)
    assert torch.allclose(torch.cat([y1, y2], 1), full, atol=1e-4, rtol=1e-4) and torch.allclose(h2, hfull, atol=1e-4, rtol=1e-4)


def test_step_equals_the_scan_one_token_at_a_time():
    x, dt, A, Bm, Cm, D = _inputs(T=7)
    y, h = ssd_scan_reference(x, dt, A, Bm, Cm, D, 4)
    state, ys = torch.zeros_like(h), []
    for t in range(7):
        yt, state = ssd_step_reference(state, x[:, t], dt[:, t], A, Bm[:, t], Cm[:, t], D)
        ys.append(yt)
    assert torch.allclose(torch.stack(ys, 1), y, atol=1e-4, rtol=1e-4) and torch.allclose(state, h, atol=1e-4, rtol=1e-4)


DOC_LAYOUTS = [(5, 3, 5), (13,), (1, 1, 11), (4, 4, 5), (2, 9, 2), (12, 1)]


def _doc_ids(lens, B=2):
    row = torch.cat([torch.full((n,), i) for i, n in enumerate(lens)])
    return row.unsqueeze(0).expand(B, -1).contiguous()


@pytest.mark.parametrize("chunk", [4, 8])
@pytest.mark.parametrize("lens", DOC_LAYOUTS)
def test_doc_ids_reset_the_state_exactly(chunk, lens):
    """Boundaries inside a chunk, exactly at a chunk edge, several in one chunk, a document longer
    than a chunk: packed equals each document alone, and equals the sequential definition."""
    args = _inputs(T=sum(lens))
    ids = _doc_ids(lens)
    y, h = ssd_scan_reference(*args, chunk, doc_ids=ids)
    x, dt, A, Bm, Cm, D = args
    parts, at = [], 0
    for n in lens:
        sl = slice(at, at + n)
        yi, hi = ssd_scan_reference(x[:, sl], dt[:, sl], A, Bm[:, sl], Cm[:, sl], D, chunk)
        parts.append(yi)
        at += n
    assert torch.allclose(y, torch.cat(parts, 1), atol=1e-4, rtol=1e-4)
    assert torch.allclose(h, hi, atol=1e-4, rtol=1e-4), "final state is the last document's"
    y0, _ = _sequential(*args, doc_ids=ids)
    assert torch.allclose(y, y0, atol=1e-4, rtol=1e-4)


def test_without_doc_ids_state_leaks_across_documents():
    args = _inputs(T=13)
    y, _ = ssd_scan_reference(*args, 4)
    y_doc, _ = ssd_scan_reference(*args, 4, doc_ids=_doc_ids((5, 8)))
    assert not torch.allclose(y[:, 5:], y_doc[:, 5:], atol=1e-3)


def test_ssd_scan_falls_back_to_the_reference_off_cuda():
    args = _inputs()
    assert torch.equal(ssd_scan(*args, 4)[0], ssd_scan_reference(*args, 4)[0])


@pytest.mark.skipif(not ssm.HAS_SSM_KERNEL, reason="needs CUDA and the kernels-community/mamba-ssm build")
class TestKernelVsReference:
    """Untestable on a CUDA-less machine; run on a GPU. bf16 kernels vs the float32 reference."""

    def _cuda(self, args, dtype):
        x, dt, A, Bm, Cm, D = (t.cuda() for t in args)
        return x.to(dtype), dt, A, Bm.to(dtype), Cm.to(dtype), D

    @pytest.mark.parametrize("lens", [(13,), (5, 3, 5), (1, 1, 11)])
    def test_scan(self, lens):
        args = _inputs(T=sum(lens), H=8, P=16, N=16, G=2)
        ids = _doc_ids(lens).cuda()
        got, hgot = ssd_scan(*self._cuda(args, torch.bfloat16), 8, doc_ids=ids)
        ref, href = ssd_scan_reference(*self._cuda(args, torch.bfloat16), 8, doc_ids=ids)
        assert torch.allclose(got.float(), ref.float(), atol=5e-2, rtol=5e-2)
        assert torch.allclose(hgot, href, atol=5e-2, rtol=5e-2)

    def test_step(self):
        x, dt, A, Bm, Cm, D = self._cuda(_inputs(T=1, H=8, P=16, N=16, G=2), torch.bfloat16)
        state = torch.randn(2, 8, 16, 16, device="cuda")
        y, h = ssm.ssd_step(state.clone(), x[:, 0], dt[:, 0], A, Bm[:, 0], Cm[:, 0], D)
        yr, hr = ssd_step_reference(state.clone(), x[:, 0], dt[:, 0], A, Bm[:, 0], Cm[:, 0], D)
        assert torch.allclose(y.float(), yr.float(), atol=5e-2, rtol=5e-2) and torch.allclose(h.float(), hr, atol=5e-2, rtol=5e-2)


# -----------------------------------------------------------------------------
# the mixer

def _mamba(**kw):
    spec = _mamba2_mixer(**kw).params
    m = Mamba2(n_embd=32, norm=RMS_NORM_INSTANCE(), **spec)
    m.bind_layer(0)
    m.init_weights()
    return m


def RMS_NORM_INSTANCE():
    from modelcore.components.norm import RMSNorm
    return RMSNorm(None)


def test_init_follows_the_reference():
    m = Mamba2(n_embd=32, norm=RMS_NORM_INSTANCE(), **_mamba2_mixer(head_dim=4, d_state=8).params)
    m.bind_layer(0)
    m.init_weights()
    dt = torch.nn.functional.softplus(m.dt_bias)
    assert (dt >= 1e-4 - 1e-7).all() and (dt <= 0.1 + 1e-6).all()
    A = torch.exp(m.A_log)
    assert (A >= 1 - 1e-5).all() and (A <= 16 + 1e-4).all()
    assert torch.equal(m.D, torch.ones_like(m.D)) and not m.out_proj.weight.any()
    assert m.n_head == 16 and m.conv_dim == 64 + 2 * 8
    assert m.in_proj.weight.shape == (2 * 64 + 2 * 8 + 16, 32)


def test_layer_spec_state_and_decode_cheaper_than_train():
    spec = _mamba().layer_spec()
    H, P, N, K = 4, 16, 16, 4
    assert spec.state_elems == H * P * N + (K - 1) * (64 + 2 * N)
    assert spec.decode_flops_per_token < spec.fwd_flops_per_token


def test_a_changed_token_reaches_only_later_outputs_and_the_mixer_is_causal():
    torch.manual_seed(0)
    m = _mamba()
    for p in m.parameters():
        torch.nn.init.normal_(p, std=0.2)
    x = torch.randn(1, 12, 32)
    x2 = x.clone()
    x2[:, 7] += 1.0
    with torch.no_grad():
        a, b = m(x, None, None), m(x2, None, None)
    assert torch.allclose(a[:, :7], b[:, :7], atol=1e-6) and not torch.allclose(a[:, 7:], b[:, 7:])


@pytest.mark.parametrize("prefill", [1, 3, 8, 11])
def test_cached_steps_equal_a_full_pass_through_the_mixer(prefill):
    torch.manual_seed(1)
    m = _mamba(chunk_size=4)
    for p in m.parameters():
        torch.nn.init.normal_(p, std=0.2)
    x = torch.randn(2, 11, 32)
    cache = KVCache(2, 0, 11, 0, 0, torch.device("cpu"), torch.float32)
    with torch.no_grad():
        full = m(x, None, None)
        outs = [m(x[:, :prefill], None, cache)] + [m(x[:, t:t + 1], None, cache) for t in range(prefill, 11)]
    assert torch.allclose(torch.cat(outs, 1), full, atol=1e-4, rtol=1e-4)
    assert cache.state["mamba2.0.ssm"].shape == (2, 4, 16, 16) and cache.state["mamba2.0.conv"].shape == (2, 3, 64 + 32)


def test_the_mixer_packed_with_doc_args_equals_each_document_alone():
    torch.manual_seed(2)
    m = _mamba(chunk_size=4)
    for p in m.parameters():
        torch.nn.init.normal_(p, std=0.2)
    lens = (6, 3, 7)
    idx = torch.randint(2, 50, (1, sum(lens)))
    at = 0
    for n in lens:
        idx[0, at] = 1
        at += n
    x = torch.randn(1, sum(lens), 32)
    with torch.no_grad():
        packed = m(x, idx, None, doc_args=build_doc_args(idx, 1))
        parts, at = [], 0
        for n in lens:
            parts.append(m(x[:, at:at + n], idx[:, at:at + n], None))
            at += n
    assert torch.allclose(packed, torch.cat(parts, 1), atol=1e-4, rtol=1e-4)


# -----------------------------------------------------------------------------
# validation, roles, stats

def _errors(manager, config):
    return [f"{e.path}: {e.message}" for e in manager.validate_config(config).errors]


@pytest.mark.parametrize("change,needle", [
    ({"head_dim": 0}, "head_dim"), ({"chunk_size": 0}, "chunk_size"), ({"kernel_size": 0}, "kernel_size"),
    ({"head_dim": 48}, "divisible by head_dim"), ({"n_groups": 3}, "divisible by n_groups"),
    ({"dt_min": 0.5}, "dt_min <= dt_max"), ({"dt_init_floor": 0}, "dt_init_floor"),
    ({"A_init_min": 0}, "A_init_min"), ({"A_init_min": 20}, "A_init_min"),
])
def test_mamba2_params_are_validated(manager, change, needle):
    config = _mixed_like(["mamba2"])
    config.body.params["blocks"][0].params["mixer"].params.update(change)
    assert any(needle in e for e in _errors(manager, config)), _errors(manager, config)


def test_every_mamba2_param_is_required(manager):
    config = _mixed_like(["mamba2"])
    del config.body.params["blocks"][0].params["mixer"].params["dt_max"]
    assert any("missing required param 'dt_max'" in e for e in _errors(manager, config))


def test_mamba2_needs_no_rope_or_kv(manager):
    config = FLAVORS["mamba2_only"]()
    assert "rope" not in config.shared
    stats = manager.stats(config)
    assert stats.kv_cache_spec["num_kv_slots"] == 0 and stats.kv_bytes_per_token() == 0
    # per layer: 8 heads x 16 x 16 state + 3 x (128 + 2*16) conv inputs
    assert stats.state_elems_per_row == 3 * (8 * 16 * 16 + 3 * 160)


def test_ssm_and_conv_params_get_their_own_adamw_groups_and_no_weight_decay(manager):
    model = build(manager, FLAVORS["mamba2_only"]())
    roles = collect_param_roles(model)
    assert {n.rsplit(".", 1)[-1] for n, p in model.named_parameters() if id(p) in {id(q) for q in roles["ssm"]}} == \
        {"A_log", "dt_bias", "D"}
    assert {n.rsplit(".", 1)[-1] for n, p in model.named_parameters() if id(p) in {id(q) for q in roles["conv"]}} == \
        {"conv_weight", "conv_bias"}
    optimizer = manager.create_optimizer(model, OptimizerHparams(ssm_lr=0.03))
    kinds = {(g["kind"], g["weight_decay"]) for g in optimizer.param_groups[-2:]}
    assert kinds == {("adamw", 0.0)}
    assert optimizer.param_groups[-1]["lr"] == pytest.approx(0.03 * (64 / 768) ** -0.5)


def test_hybrid_state_is_constant_and_attention_kv_is_only_the_attention_layers(manager):
    hybrid = manager.stats(FLAVORS["hybrid_mamba2_attn"]())
    assert hybrid.kv_cache_spec["num_kv_slots"] == 2 and hybrid.state_elems_per_row > 0
    assert hybrid.kv_bytes_per_token() == 2 * 2 * 2 * 32 * 4  # 2 slots x (K,V) x 2 heads x 32 dim x fp32
