import torch
import torch.nn as nn
import torch.nn.functional as F

from modelcore.components.linear import Linear


class MLP(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.n_embd = n_embd
        self.c_fc = Linear(n_embd, 4 * n_embd, bias=False)
        self.c_proj = Linear(4 * n_embd, n_embd, bias=False)

    @torch.no_grad()
    def init_weights(self):
        s = 3**0.5 * self.n_embd**-0.5
        torch.nn.init.uniform_(self.c_fc.weight, -s * 0.4, s * 0.4)  # 0.4x init scale for c_fc
        torch.nn.init.zeros_(self.c_proj.weight)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class SwiGLUMLP(nn.Module):
    """Llama-style gated MLP: two parallel projections (gate, up) combined via SiLU-gating, then
    projected back down. hidden_dim follows the standard Llama derivation: 2/3 of the usual 4x
    expansion (to keep matmul FLOPs roughly matched to a plain 4x MLP despite the extra gate
    projection), rounded up to a multiple of multiple_of for clean tiling.

    All three projections are modelcore.components.linear.Linear, so they default to
    PARAM_ROLES role "matrix" (see modelcore.roles) with no declaration needed here."""

    def __init__(self, n_embd, multiple_of=256):
        super().__init__()
        self.n_embd = n_embd
        hidden_dim = int(2 * (4 * n_embd) / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.hidden_dim = hidden_dim
        self.gate_proj = Linear(n_embd, hidden_dim, bias=False)
        self.up_proj = Linear(n_embd, hidden_dim, bias=False)
        self.down_proj = Linear(hidden_dim, n_embd, bias=False)

    @torch.no_grad()
    def init_weights(self):
        s = 3**0.5 * self.n_embd**-0.5  # sqrt(3) multiplier makes sure Uniform achieves the same std as Normal
        torch.nn.init.uniform_(self.gate_proj.weight, -s, s)
        torch.nn.init.uniform_(self.up_proj.weight, -s, s)
        torch.nn.init.zeros_(self.down_proj.weight)  # projection back down starts at zero, like GPT's mlp.c_proj

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
