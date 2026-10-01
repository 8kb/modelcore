"""
The Mamba-3 mixer, SISO (Lahoti et al., "Mamba-3: Improved Sequence Modeling using State Space
Principles"; reference: state-spaces/mamba, mamba_ssm/modules/mamba3.py and
ops/triton/mamba3/mamba3_siso_step.py, which this follows). Three changes from Mamba-2:

- exponential-trapezoidal discretization: h_t = a_t h_{t-1} + b_t (k_{t-1} x_{t-1}) + g_t (k_t x_t)
  with a_t = exp(A_t dt_t), g_t = lam_t dt_t, b_t = (1 - lam_t) dt_t a_t, lam_t = sigmoid(...): the
  state sees the previous token too, an implicit width-2 convolution that replaces Mamba-2's
  short conv (there is none here). A_t is data-dependent, per head and token.
- a complex-valued state, implemented as a data-dependent rotation of B and C: both are rotated
  by the same cumulative angle sum_i tanh(w_i) * pi * dt_i, so only the relative rotation between
  token t and token s survives in C_t . B_s. This is NOT positional RoPE (shared.rope): the angles
  come from the input, the recurrence already orders tokens, and a Mamba-3 layer needs no rope
  in `shared`. The rotation reuses components/rope.py's apply_rotary_emb on the first
  2 * num_rope_angles dims of B/C (the rest are left unrotated).
- BC normalisation (a norm on B and C, like QK-norm) and a learnable per-head bias on each.

Scan: rather than a new kernel, the recurrence is folded into the plain SSD scan of
modelcore.kernels.ssm. With c_t = dt_t (1 - lam_t) and s_t = h_t + c_{t+1} k_t x_t, s obeys
s_t = a_t s_{t-1} + (g_t + c_{t+1}) k_t x_t, an ordinary scan whose input weight is g_t + c_{t+1};
then h_t = s_t - c_{t+1} k_t x_t, so y_t = q_t . s_t - c_{t+1} (q_t . k_t) x_t. c_{t+1} is 0 at the
last token of a call and of a document (there is no next token to fold in), which also makes the
scan's final state the true h. Streaming continues from the cache's h, k_{t-1}, x_{t-1} by
starting the scan at s_{-1} = h + c_0 k_{-1} x_{-1}. Decode (one token) uses the direct
recurrence, and tests check the two agree.

`mimo_rank` is in the spec from day one (1 = SISO) so MIMO's parameter shapes don't change a config
later; the validator rejects > 1 until it is implemented.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from modelcore.catalog import register_component
from modelcore.components.contracts import BaseMixer, is_int
from modelcore.components.linear import Linear
from modelcore.components.rope import apply_rotary_emb
from modelcore.config.spec import RecurrentLayerSpec
from modelcore.kernels.ssm import ssd_scan_decay


def heavy_tail_activation(x):
    """f(x) = 1 + x for x >= 0, 1 / (1 - x) for x < 0: positive, continuous and differentiable at 0.
    The reference uses it for the data-dependent A (better stability at high learning rates)."""
    return x.clamp_min(0) + torch.reciprocal(1 - x.clamp_max(0))


def num_rope_angles(d_state, rope_fraction):
    split = int(d_state * rope_fraction)
    if split % 2:
        split -= 1
    return split // 2


def _validate_mamba3(params, ctx):
    errors = []
    for name in ("d_state", "head_dim", "expand", "n_groups", "chunk_size", "mimo_rank"):
        v = params.get(name)
        if not is_int(v) or v < 1:
            errors.append(f"{name} must be a positive integer, got {v!r}")
    if is_int(params.get("mimo_rank")) and params["mimo_rank"] > 1:
        errors.append("mimo_rank > 1 (MIMO) is not implemented yet; use mimo_rank=1 (SISO)")
    if params.get("rope_fraction") not in (0.5, 1.0):
        errors.append(f"rope_fraction must be 0.5 or 1.0, got {params.get('rope_fraction')!r}")
    elif is_int(params.get("d_state")) and params["d_state"] >= 1 and num_rope_angles(params["d_state"], params["rope_fraction"]) < 1:
        errors.append(f"d_state {params['d_state']} is too small to rotate (need at least one angle)")
    n_embd = ctx.get("n_embd")
    expand, head_dim, n_groups = params.get("expand"), params.get("head_dim"), params.get("n_groups")
    if all(is_int(v) and v >= 1 for v in (expand, head_dim, n_groups)) and n_embd is not None:
        if (expand * n_embd) % head_dim:
            errors.append(f"expand * n_embd ({expand * n_embd}) must be divisible by head_dim ({head_dim})")
        elif ((expand * n_embd) // head_dim) % n_groups:
            errors.append(f"n_head ({(expand * n_embd) // head_dim}) must be divisible by n_groups ({n_groups})")
    dt_min, dt_max, floor = params.get("dt_min"), params.get("dt_max"), params.get("dt_init_floor")
    if not (isinstance(dt_min, (int, float)) and isinstance(dt_max, (int, float)) and 0 < dt_min <= dt_max):
        errors.append(f"need 0 < dt_min <= dt_max, got dt_min={dt_min!r}, dt_max={dt_max!r}")
    if not (isinstance(floor, (int, float)) and floor > 0):
        errors.append(f"dt_init_floor must be positive, got {floor!r}")
    a_floor = params.get("A_floor")
    if not (isinstance(a_floor, (int, float)) and a_floor > 0):
        errors.append(f"A_floor must be positive, got {a_floor!r}")
    return errors


@register_component("mamba3", needs=("n_embd", "norm"), validate=_validate_mamba3)
class Mamba3(BaseMixer):
    """Mamba-3 (SISO). d_inner = expand * n_embd in n_head = d_inner // head_dim heads; B and C (width
    d_state) are shared across the heads of one of n_groups groups, normalised with the config's shared
    (parameterless) norm -- the reference's BC norms carry a learnable gain, which the norms rule here
    drops -- then given a learnable per-head bias (init 1) and rotated. Per token:

        z, x, B, C, dd_dt, dd_A, trap, angles = in_proj(u)
        dt = softplus(dd_dt + dt_bias);  A = -max(heavy_tail(dd_A), A_floor);  lam = sigmoid(trap)
        y = (SSM(x, dt, A, lam, rotated B, C) + D * x) * silu(z);   out_proj(y)

    (the reference's optional output-projection norm is not implemented). Params: in_proj/out_proj
    Linear (matrix); dt_bias, B_bias, C_bias, D role "ssm" (AdamW, no weight decay). Inference state
    per row, in the cache: the SSM state (n_head, head_dim, d_state), the previous token's rotated B
    (n_head, d_state) and x (n_head, head_dim), and the running angle (n_head, num_rope_angles).
    Intra-document masking resets the state, and the previous-token term, at a document boundary;
    the rotation angle is not reset (only relative angles matter, as with RoPE)."""
    PARAM_ROLES = {"dt_bias": "ssm", "B_bias": "ssm", "C_bias": "ssm", "D": "ssm"}

    def __init__(self, n_embd, norm, d_state, head_dim, expand, n_groups, mimo_rank, rope_fraction, chunk_size,
                 dt_min, dt_max, dt_init_floor, A_floor):
        super().__init__()
        self.n_embd, self.norm = n_embd, norm
        self.d_state, self.head_dim, self.n_groups, self.mimo_rank = d_state, head_dim, n_groups, mimo_rank
        self.chunk_size, self.A_floor = chunk_size, A_floor
        self.d_inner = expand * n_embd
        assert self.d_inner % head_dim == 0
        self.n_head = self.d_inner // head_dim
        assert self.n_head % n_groups == 0
        self.n_angles = num_rope_angles(d_state, rope_fraction)
        self._init = dict(dt_min=dt_min, dt_max=dt_max, dt_init_floor=dt_init_floor)
        H, N, R = self.n_head, d_state, mimo_rank
        # order: [z, x, B, C, dd_dt, dd_A, trap, angles]
        self.in_proj = Linear(n_embd, 2 * self.d_inner + 2 * N * n_groups * R + 3 * H + self.n_angles, bias=False)
        self.dt_bias = nn.Parameter(torch.empty(H))  # fake init, real init in init_weights()
        self.B_bias = nn.Parameter(torch.empty(H, R, N))
        self.C_bias = nn.Parameter(torch.empty(H, R, N))
        self.D = nn.Parameter(torch.empty(H))
        self.out_proj = Linear(self.d_inner, n_embd, bias=False)

    def bind_layer(self, layer_idx):
        self.layer_idx = layer_idx

    @torch.no_grad()
    def init_weights(self):
        s = 3**0.5 * self.n_embd**-0.5
        torch.nn.init.uniform_(self.in_proj.weight, -s, s)
        i = self._init
        u = torch.rand(self.n_head, device=self.dt_bias.device)
        dt = torch.exp(u * (math.log(i["dt_max"]) - math.log(i["dt_min"])) + math.log(i["dt_min"]))
        dt = dt.clamp(min=i["dt_init_floor"])
        self.dt_bias.copy_(dt + torch.log(-torch.expm1(-dt)))  # inverse of softplus
        torch.nn.init.ones_(self.B_bias)
        torch.nn.init.ones_(self.C_bias)
        torch.nn.init.ones_(self.D)
        torch.nn.init.zeros_(self.out_proj.weight)

    def layer_spec(self):
        H, P, N, S, Q = self.n_head, self.head_dim, self.d_state, self.n_angles, self.chunk_size
        state = H * P * N + H * N + H * P + H * S
        scan = 6 * H * P * N + 8 * H * N  # state update + readout; the rotation of B and C
        return RecurrentLayerSpec("mamba3", state_elems=state, fwd_flops_per_token=scan + 2 * Q * H * (N + P),
                                  decode_flops_per_token=scan)

    def _rotate(self, t, angle):
        """Rotate the first 2 * n_angles dims of t (B, T, H, N) by angle (B, T, H, n_angles)."""
        r = 2 * self.n_angles
        cos, sin = angle.cos(), angle.sin()
        return torch.cat([apply_rotary_emb(t[..., :r], cos, sin), t[..., r:]], dim=-1)

    def forward(self, x, idx, cache, bus=None, doc_args=None):
        Bsz, T, _ = x.shape
        H, P, N, G, S, R = self.n_head, self.head_dim, self.d_state, self.n_groups, self.n_angles, self.mimo_rank
        z, xs, Bp, Cp, dd_dt, dd_A, trap, ang = self.in_proj(x).split(
            [self.d_inner, self.d_inner, N * G * R, N * G * R, H, H, H, S], dim=-1)
        z, xs = z.view(Bsz, T, H, P).float(), xs.view(Bsz, T, H, P).float()

        # B, C: (r g n) -> norm -> per-head + bias -> rotate.  R == 1 until MIMO.
        Bp, Cp = self.norm(Bp.view(Bsz, T, G, N)), self.norm(Cp.view(Bsz, T, G, N))
        Bh = Bp.float().repeat_interleave(H // G, dim=2) + self.B_bias[:, 0].float()  # (B, T, H, N)
        Ch = Cp.float().repeat_interleave(H // G, dim=2) + self.C_bias[:, 0].float()

        A = (-heavy_tail_activation(dd_A.float())).clamp(max=-self.A_floor)            # (B, T, H) < 0
        dt = F.softplus(dd_dt.float() + self.dt_bias.float())
        lam = torch.sigmoid(trap.float())
        gamma, c = lam * dt, dt * (1 - lam)

        key = f"mamba3.{self.layer_idx}"
        st = None if cache is None else cache.state.get(key + ".ssm")
        angle0 = torch.zeros(Bsz, H, S, device=x.device) if st is None else cache.state[key + ".angle"]

        # cumulative rotation angle per (token, head, pair), kept in [0, 2 pi)
        dang = (torch.tanh(ang.float()) * math.pi).unsqueeze(2) * dt.unsqueeze(-1)      # (B, T, H, S)
        angle = torch.remainder(torch.cumsum(dang, dim=1) + angle0.unsqueeze(1), 2 * math.pi)
        k, q = self._rotate(Bh, angle), self._rotate(Ch, angle)

        if T == 1 and st is not None:  # decode: the direct recurrence
            k_prev, v_prev = cache.state[key + ".k"], cache.state[key + ".v"]
            k0, q0, x0, dt0, g0, c0 = k[:, 0], q[:, 0], xs[:, 0], dt[:, 0], gamma[:, 0], c[:, 0]
            alpha = torch.exp(A[:, 0] * dt0)[..., None, None]
            beta = (alpha[..., 0, 0] * c0)[..., None, None]
            h = st * alpha + beta * (v_prev[..., None] * k_prev[:, :, None, :]) + g0[..., None, None] * (x0[..., None] * k0[:, :, None, :])
            y = (torch.einsum("bhpn,bhn->bhp", h, q0) + self.D.float().view(1, H, 1) * x0).unsqueeze(1)
            final = h
        else:
            # the previous token (or document) hands its input to this one through c_{t+1}
            c_next = torch.cat([c[:, 1:], torch.zeros_like(c[:, :1])], dim=1)
            if doc_args is not None and cache is None:
                ids = doc_args.doc_ids
                same = torch.cat([ids[:, 1:] == ids[:, :-1], torch.zeros_like(ids[:, :1], dtype=torch.bool)], dim=1)
                c_next = c_next * same.unsqueeze(-1)
                doc_ids = ids
            else:
                doc_ids = None
            s0 = None
            if st is not None:  # continue a stream: fold the previous token into the initial state
                s0 = st + c[:, 0, :, None, None] * (cache.state[key + ".v"][..., None] * cache.state[key + ".k"][:, :, None, :])
            u = gamma + c_next
            y, final = ssd_scan_decay(xs * u.unsqueeze(-1), A * dt, k, q, self.chunk_size, s0, doc_ids)
            y = y.float() - (c_next * (q * k).sum(-1)).unsqueeze(-1) * xs + self.D.float().view(1, 1, H, 1) * xs

        if cache is not None:
            cache.state[key + ".ssm"], cache.state[key + ".angle"] = final, angle[:, -1]
            cache.state[key + ".k"], cache.state[key + ".v"] = k[:, -1], xs[:, -1]

        y = (y * F.silu(z)).reshape(Bsz, T, self.d_inner)
        return self.out_proj(y.to(x.dtype))
