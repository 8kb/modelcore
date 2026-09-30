"""
The Mamba-2 mixer (Dao & Gu, "Transformers are SSMs"): in_proj -> causal conv over [x, B, C] ->
SSD scan -> gated norm -> out_proj. The scan itself lives in modelcore.kernels.ssm (pure-PyTorch
chunked reference, mamba_ssm's kernels on CUDA).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from modelcore.catalog import register_component
from modelcore.components.contracts import BaseMixer
from modelcore.components.conv import causal_depthwise_conv
from modelcore.components.linear import Linear
from modelcore.config.spec import RecurrentLayerSpec
from modelcore.kernels.ssm import ssd_scan, ssd_step


def _is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def _validate_mamba2(params, ctx):
    errors = []
    for name in ("d_state", "head_dim", "expand", "n_groups", "kernel_size", "chunk_size"):
        v = params.get(name)
        if not _is_int(v) or v < 1:
            errors.append(f"{name} must be a positive integer, got {v!r}")
    n_embd = ctx.get("n_embd")
    expand, head_dim, n_groups = params.get("expand"), params.get("head_dim"), params.get("n_groups")
    if all(_is_int(v) and v >= 1 for v in (expand, head_dim, n_groups)) and n_embd is not None:
        if (expand * n_embd) % head_dim:
            errors.append(f"expand * n_embd ({expand * n_embd}) must be divisible by head_dim ({head_dim})")
        elif ((expand * n_embd) // head_dim) % n_groups:
            errors.append(f"n_head ({(expand * n_embd) // head_dim}) must be divisible by n_groups ({n_groups})")
    dt_min, dt_max, floor = params.get("dt_min"), params.get("dt_max"), params.get("dt_init_floor")
    if not (isinstance(dt_min, (int, float)) and isinstance(dt_max, (int, float)) and 0 < dt_min <= dt_max):
        errors.append(f"need 0 < dt_min <= dt_max, got dt_min={dt_min!r}, dt_max={dt_max!r}")
    if not (isinstance(floor, (int, float)) and floor > 0):
        errors.append(f"dt_init_floor must be positive, got {floor!r}")
    a_min, a_max = params.get("A_init_min"), params.get("A_init_max")
    if not (isinstance(a_min, (int, float)) and isinstance(a_max, (int, float)) and 0 < a_min <= a_max):
        errors.append(f"need 0 < A_init_min <= A_init_max, got {a_min!r}, {a_max!r}")
    return errors


@register_component("mamba2", needs=("n_embd", "norm"), validate=_validate_mamba2)
class Mamba2(BaseMixer):
    """Mamba-2. d_inner = expand * n_embd is split into n_head = d_inner // head_dim heads of width
    head_dim; each head keeps a (head_dim, d_state) state, B and C are shared across the heads of
    one of n_groups groups. Per token:

        z, xBC, dt = in_proj(x)                        xBC = [x (d_inner), B, C (n_groups * d_state each)]
        xBC = silu(causal_conv(xBC) + bias)            kernel_size taps, depthwise
        dt = softplus(dt + dt_bias);  A = -exp(A_log)  per head
        y = SSD(x, dt, A, B, C) + D * x                the scan, see modelcore.kernels.ssm
        out_proj(norm(y * silu(z)))                    gated norm

    The norm is the config's shared, parameterless one (the reference's RMSNormGated has a learnable
    gain; norms here never do -- see AGENTS.md). Inference state per row: the conv's last
    kernel_size-1 inputs and the (n_head, head_dim, d_state) SSM state, in the cache; no KV, no
    positional encoding (the recurrence orders tokens). Intra-document masking resets both.

    Params: in_proj/out_proj are Linear (role matrix); conv weight/bias are role "conv"; A_log,
    dt_bias, D are per-head vectors of role "ssm" (AdamW, no weight decay). Init follows the
    reference: dt log-uniform in [dt_min, dt_max] (floored at dt_init_floor) stored through the
    inverse softplus, A uniform in [A_init_min, A_init_max], D = 1; out_proj is zero like every
    residual projection here."""
    PARAM_ROLES = {"conv_weight": "conv", "conv_bias": "conv", "A_log": "ssm", "dt_bias": "ssm", "D": "ssm"}

    def __init__(self, n_embd, norm, d_state, head_dim, expand, n_groups, kernel_size, chunk_size,
                 dt_min, dt_max, dt_init_floor, A_init_min, A_init_max):
        super().__init__()
        self.n_embd, self.norm = n_embd, norm
        self.d_state, self.head_dim, self.n_groups = d_state, head_dim, n_groups
        self.kernel_size, self.chunk_size = kernel_size, chunk_size
        self.d_inner = expand * n_embd
        assert self.d_inner % head_dim == 0
        self.n_head = self.d_inner // head_dim
        assert self.n_head % n_groups == 0
        self.conv_dim = self.d_inner + 2 * n_groups * d_state
        self._init = dict(dt_min=dt_min, dt_max=dt_max, dt_init_floor=dt_init_floor,
                          A_init_min=A_init_min, A_init_max=A_init_max)
        self.in_proj = Linear(n_embd, 2 * self.d_inner + 2 * n_groups * d_state + self.n_head, bias=False)
        self.conv_weight = nn.Parameter(torch.empty(self.conv_dim, kernel_size))  # fake init, real in init_weights()
        self.conv_bias = nn.Parameter(torch.empty(self.conv_dim))
        self.dt_bias = nn.Parameter(torch.empty(self.n_head))
        self.A_log = nn.Parameter(torch.empty(self.n_head))
        self.D = nn.Parameter(torch.empty(self.n_head))
        self.out_proj = Linear(self.d_inner, n_embd, bias=False)

    def bind_layer(self, layer_idx):
        self.layer_idx = layer_idx

    @torch.no_grad()
    def init_weights(self):
        s = 3**0.5 * self.n_embd**-0.5
        torch.nn.init.uniform_(self.in_proj.weight, -s, s)
        bound = self.kernel_size**-0.5
        torch.nn.init.uniform_(self.conv_weight, -bound, bound)
        torch.nn.init.uniform_(self.conv_bias, -bound, bound)
        i = self._init
        u = torch.rand(self.n_head, device=self.dt_bias.device)
        dt = torch.exp(u * (math.log(i["dt_max"]) - math.log(i["dt_min"])) + math.log(i["dt_min"]))
        dt = dt.clamp(min=i["dt_init_floor"])
        self.dt_bias.copy_(dt + torch.log(-torch.expm1(-dt)))  # inverse of softplus
        A = torch.empty(self.n_head, device=self.A_log.device).uniform_(i["A_init_min"], i["A_init_max"])
        self.A_log.copy_(torch.log(A))
        torch.nn.init.ones_(self.D)
        torch.nn.init.zeros_(self.out_proj.weight)

    def layer_spec(self):
        H, P, N, K, Q = self.n_head, self.head_dim, self.d_state, self.kernel_size, self.chunk_size
        state = H * P * N + (K - 1) * self.conv_dim
        conv = 2 * K * self.conv_dim
        # per token: state update + readout (~6 flops per state element); training/prefill's chunked
        # form adds the within-chunk quadratic term. An estimate, like the attention term above it.
        scan = 6 * H * P * N
        return RecurrentLayerSpec("mamba2", state_elems=state, fwd_flops_per_token=conv + scan + 2 * Q * H * (N + P),
                                  decode_flops_per_token=conv + scan)

    def forward(self, x, idx, cache, bus=None, doc_args=None):
        B, T, _ = x.shape
        H, P, N, G = self.n_head, self.head_dim, self.d_state, self.n_groups
        z, xBC, dt = self.in_proj(x).split([self.d_inner, self.conv_dim, H], dim=-1)

        conv_key, ssm_key = f"mamba2.{self.layer_idx}.conv", f"mamba2.{self.layer_idx}.ssm"
        conv_state = None if cache is None else cache.state.get(conv_key)
        ssm_state = None if cache is None else cache.state.get(ssm_key)
        doc_ids = doc_args.doc_ids if (doc_args is not None and cache is None) else None

        xBC, new_conv = causal_depthwise_conv(xBC, self.conv_weight, conv_state, doc_ids)
        xBC = F.silu(xBC + self.conv_bias.to(xBC.dtype))
        xs, Bm, Cm = xBC.split([self.d_inner, G * N, G * N], dim=-1)
        xs, Bm, Cm = xs.view(B, T, H, P), Bm.view(B, T, G, N), Cm.view(B, T, G, N)
        dt = F.softplus(dt.float() + self.dt_bias.float())
        A = -torch.exp(self.A_log.float())

        if T == 1 and ssm_state is not None:
            y, new_ssm = ssd_step(ssm_state, xs[:, 0], dt[:, 0], A, Bm[:, 0], Cm[:, 0], self.D)
            y = y.unsqueeze(1)
        else:
            y, new_ssm = ssd_scan(xs, dt, A, Bm, Cm, self.D, self.chunk_size, ssm_state, doc_ids)
        if cache is not None:
            cache.state[conv_key], cache.state[ssm_key] = new_conv, new_ssm

        y = self.norm(y.reshape(B, T, self.d_inner) * F.silu(z))
        return self.out_proj(y)
