"""
The Mamba-3 (SISO) mixer, checked against the paper's recurrence written out token by token, with
the rotation built from explicit 2x2 blocks (nothing shared with the implementation but the
parameters). Cached/packed/whole-model behaviour is covered for every recurrent flavor in
test_recurrent.py.

python -m pytest modelcore/tests/test_mamba3.py -v
"""
import math

import pytest
import torch
import torch.nn.functional as F

from modelcore import KVCache, OptimizerHparams
from modelcore.components.mamba3 import Mamba3, heavy_tail_activation, num_rope_angles
from modelcore.components.norm import RMSNorm
from modelcore.roles import collect_param_roles
from modelcore.tests.conftest import FLAVORS, _mamba3_mixer, _mixed_like, build


def _mixer(n_embd=32, **kw):
    m = Mamba3(n_embd=n_embd, norm=RMSNorm(None), **_mamba3_mixer(**kw).params)
    m.bind_layer(0)
    m.init_weights()
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for p in m.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.3)
    return m


def _rot(t, angle, n_angles):
    """Rotate the first 2*n_angles dims of t (H, N) by angle (H, S): explicit blocks, pair i being
    (dim i, dim i + S) with y1 = x1 cos + x2 sin, y2 = -x1 sin + x2 cos (the repo's RoPE convention)."""
    H, N = t.shape
    S = n_angles
    out = t.clone()
    for h in range(H):
        for i in range(S):
            th = angle[h, i]
            R = torch.tensor([[math.cos(th), math.sin(th)], [-math.sin(th), math.cos(th)]])
            out[h, [i, i + S]] = R @ t[h, [i, i + S]]
    return out


def _paper_recurrence(m, u):
    """h_t = a_t h_{t-1} + b_t k_{t-1} x_{t-1} + g_t k_t x_t; y_t = h_t . q_t + D x_t; * silu(z); out_proj."""
    Bsz, T, _ = u.shape
    H, P, N, G, S = m.n_head, m.head_dim, m.d_state, m.n_groups, m.n_angles
    outs = []
    for b in range(Bsz):
        proj = u[b] @ m.in_proj.weight.T
        z, x, Bp, Cp, dd_dt, dd_A, trap, ang = proj.split([m.d_inner, m.d_inner, N * G, N * G, H, H, H, S], dim=-1)
        h = torch.zeros(H, P, N)
        k_prev, x_prev, angle = torch.zeros(H, N), torch.zeros(H, P), torch.zeros(H, S)
        ys = []
        for t in range(T):
            dt = F.softplus(dd_dt[t] + m.dt_bias)
            A = -torch.clamp(heavy_tail_activation(dd_A[t]), min=m.A_floor)
            lam = torch.sigmoid(trap[t])
            angle = angle + torch.tanh(ang[t])[None, :] * math.pi * dt[:, None]
            Bn = m.norm(Bp[t].view(1, 1, G, N))[0, 0].repeat_interleave(H // G, dim=0)
            Cn = m.norm(Cp[t].view(1, 1, G, N))[0, 0].repeat_interleave(H // G, dim=0)
            k = _rot(Bn + m.B_bias[:, 0], angle, S)
            q = _rot(Cn + m.C_bias[:, 0], angle, S)
            xt = x[t].view(H, P)
            alpha = torch.exp(A * dt)
            gamma = lam * dt
            beta = (1 - lam) * dt * alpha
            h = (alpha[:, None, None] * h + beta[:, None, None] * x_prev[:, :, None] * k_prev[:, None, :]
                 + gamma[:, None, None] * xt[:, :, None] * k[:, None, :])
            y = torch.einsum("hpn,hn->hp", h, q) + m.D[:, None] * xt
            ys.append((y * F.silu(z[t].view(H, P))).reshape(-1))
            k_prev, x_prev = k, xt
        outs.append(torch.stack(ys) @ m.out_proj.weight.T)
    return torch.stack(outs)


@pytest.mark.parametrize("kw", [dict(), dict(n_groups=2, chunk_size=4), dict(rope_fraction=1.0, d_state=8, chunk_size=3)])
def test_forward_matches_the_papers_recurrence(kw):
    m = _mixer(**kw)
    u = torch.randn(2, 11, 32, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        got, expected = m(u, None, None), _paper_recurrence(m, u)
    assert torch.allclose(got, expected, atol=1e-4, rtol=1e-4), (got - expected).abs().max()


def test_cached_decode_matches_the_papers_recurrence_step_by_step():
    m = _mixer(chunk_size=4)
    u = torch.randn(1, 9, 32, generator=torch.Generator().manual_seed(2))
    cache = KVCache(1, 0, 9, 0, 0, torch.device("cpu"), torch.float32)
    with torch.no_grad():
        outs = [m(u[:, :4], None, cache)] + [m(u[:, t:t + 1], None, cache) for t in range(4, 9)]
        expected = _paper_recurrence(m, u)
    assert torch.allclose(torch.cat(outs, 1), expected, atol=1e-4, rtol=1e-4)
    s = cache.state
    assert s["mamba3.0.ssm"].shape == (1, m.n_head, 16, 16) and s["mamba3.0.k"].shape == (1, m.n_head, 16)
    assert s["mamba3.0.v"].shape == (1, m.n_head, 16) and s["mamba3.0.angle"].shape == (1, m.n_head, m.n_angles)
    assert (s["mamba3.0.angle"] >= 0).all() and (s["mamba3.0.angle"] < 2 * math.pi).all()


def test_the_rotation_is_relative_not_positional():
    """Shifting every angle by a constant leaves the output unchanged: only C_t . B_s's relative
    rotation matters, which is why a Mamba-3 layer needs no positional RoPE."""
    m = _mixer()
    u = torch.randn(1, 8, 32, generator=torch.Generator().manual_seed(3))
    cache = KVCache(1, 0, 8, 0, 0, torch.device("cpu"), torch.float32)
    cache2 = KVCache(1, 0, 8, 0, 0, torch.device("cpu"), torch.float32)
    # a stream that starts at a nonzero angle (and with zeroed SSM/k/v state, so nothing else differs)
    for c, a in ((cache, 0.0), (cache2, 1.234)):
        c.state.update({"mamba3.0.ssm": torch.zeros(1, m.n_head, 16, 16), "mamba3.0.angle": torch.full((1, m.n_head, m.n_angles), a),
                        "mamba3.0.k": torch.zeros(1, m.n_head, 16), "mamba3.0.v": torch.zeros(1, m.n_head, 16)})
    with torch.no_grad():
        assert torch.allclose(m(u, None, cache), m(u, None, cache2), atol=1e-4, rtol=1e-4)


def test_heavy_tail_activation():
    x = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])
    assert torch.allclose(heavy_tail_activation(x), torch.tensor([0.25, 0.5, 1.0, 2.0, 4.0]))


@pytest.mark.parametrize("d_state,fraction,expected", [(128, 0.5, 32), (128, 1.0, 64), (16, 0.5, 4), (6, 0.5, 1), (5, 1.0, 2)])
def test_num_rope_angles_matches_the_reference_rule(d_state, fraction, expected):
    assert num_rope_angles(d_state, fraction) == expected


def test_init_and_shapes():
    m = Mamba3(n_embd=32, norm=RMSNorm(None), **_mamba3_mixer().params)
    m.bind_layer(0)
    m.init_weights()
    assert m.n_head == 4 and m.n_angles == 4
    assert m.in_proj.weight.shape == (2 * 64 + 2 * 16 + 3 * 4 + 4, 32)
    assert m.B_bias.shape == (4, 1, 16) and m.B_bias.eq(1).all() and m.C_bias.eq(1).all() and m.D.eq(1).all()
    dt = F.softplus(m.dt_bias)
    assert (dt >= 1e-4 - 1e-7).all() and (dt <= 0.1 + 1e-6).all()
    assert not m.out_proj.weight.any()


def test_layer_spec():
    spec = _mixer().layer_spec()
    H, P, N, S = 4, 16, 16, 4
    assert spec.state_elems == H * P * N + H * N + H * P + H * S
    assert spec.decode_flops_per_token < spec.fwd_flops_per_token


def _errors(manager, config):
    return [f"{e.path}: {e.message}" for e in manager.validate_config(config).errors]


@pytest.mark.parametrize("change,needle", [
    ({"mimo_rank": 4}, "not implemented"), ({"mimo_rank": 0}, "mimo_rank"), ({"rope_fraction": 0.3}, "rope_fraction"),
    ({"d_state": 1}, "too small"), ({"head_dim": 48}, "divisible by head_dim"), ({"n_groups": 3}, "divisible by n_groups"),
    ({"A_floor": 0}, "A_floor"), ({"dt_min": 1.0}, "dt_min <= dt_max"), ({"chunk_size": 0}, "chunk_size"),
])
def test_mamba3_params_are_validated(manager, change, needle):
    config = _mixed_like(["mamba3"])
    config.body.params["blocks"][0].params["mixer"].params.update(change)
    assert any(needle in e for e in _errors(manager, config)), _errors(manager, config)


def test_no_rope_in_shared_and_no_conv_and_no_kv(manager):
    config = FLAVORS["mamba3_only"]()
    assert "rope" not in config.shared
    model = build(manager, config)
    assert not any("conv" in k for k in model.state_dict())
    stats = manager.stats(config)
    assert stats.kv_cache_spec["num_kv_slots"] == 0
    H, P, N, S = 8, 16, 16, 4  # n_embd 64, expand 2, head_dim 16
    assert stats.state_elems_per_row == 3 * (H * P * N + H * N + H * P + H * S)


def test_roles(manager):
    model = build(manager, FLAVORS["mamba3_only"]())
    roles = collect_param_roles(model)
    names = {n.rsplit(".", 1)[-1] for n, p in model.named_parameters() if id(p) in {id(q) for q in roles["ssm"]}}
    assert names == {"dt_bias", "B_bias", "C_bias", "D"}
    assert "conv" not in roles
    optimizer = manager.create_optimizer(model, OptimizerHparams())
    assert optimizer.param_groups[-1]["weight_decay"] == 0.0 and optimizer.param_groups[-1]["kind"] == "adamw"
