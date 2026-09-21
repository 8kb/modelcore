import torch
import torch.nn as nn
import torch.nn.functional as F

from modelcore.catalog import register_component
from modelcore.components.linear import Linear

# Activation name -> function. The name is what a config says; keeping the table here (rather than
# in each class) is what makes "which nonlinearity" a config choice instead of a class choice.
def _relu2(x):
    return F.relu(x).square()


ACTIVATIONS = {
    "relu2": _relu2,
    "relu": F.relu,
    "gelu": F.gelu,
    "silu": F.silu,
}
PLAIN_ACTIVATIONS = ("relu2", "relu", "gelu", "silu")
GATED_ACTIVATIONS = ("silu", "gelu", "relu")


def _validator(allowed):
    def validate(params, ctx):
        errors = []
        activation = params.get("activation")
        if activation not in allowed:
            errors.append(f"activation must be one of {list(allowed)}, got {activation!r}")
        hidden_dim = params.get("hidden_dim")
        if not isinstance(hidden_dim, int) or isinstance(hidden_dim, bool) or hidden_dim <= 0:
            errors.append(f"hidden_dim must be a positive integer, got {hidden_dim!r}")
        return errors
    return validate


@register_component("mlp", needs=("n_embd",), validate=_validator(PLAIN_ACTIVATIONS))
class MLP(nn.Module):
    """c_proj(act(c_fc(x))). Both `activation` and `hidden_dim` are stated by the config -- there
    is no 4 * n_embd here: the rule that used to produce that number lives in whatever host layer
    materializes the tree (a config carries only concrete values, never a derivation rule)."""

    def __init__(self, n_embd, activation, hidden_dim):
        super().__init__()
        self.n_embd = n_embd
        self.hidden_dim = hidden_dim
        self.act = ACTIVATIONS[activation]
        self.c_fc = Linear(n_embd, hidden_dim, bias=False)
        self.c_proj = Linear(hidden_dim, n_embd, bias=False)

    @torch.no_grad()
    def init_weights(self):
        s = 3**0.5 * self.n_embd**-0.5
        torch.nn.init.uniform_(self.c_fc.weight, -s * 0.4, s * 0.4)  # 0.4x init scale for c_fc
        torch.nn.init.zeros_(self.c_proj.weight)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        return x


@register_component("gated_mlp", needs=("n_embd",), validate=_validator(GATED_ACTIVATIONS))
class GatedMLP(nn.Module):
    """Gated MLP (SwiGLU when activation="silu"): two parallel projections (gate, up) combined as
    act(gate(x)) * up(x), then projected back down. hidden_dim is stated by the config, not derived
    -- the Llama rule (2/3 of a 4x expansion, rounded up to a multiple) is a host-layer derivation.

    All three projections are modelcore.components.linear.Linear, so they default to
    PARAM_ROLES role "matrix" (see modelcore.roles) with no declaration needed here."""

    def __init__(self, n_embd, activation, hidden_dim):
        super().__init__()
        self.n_embd = n_embd
        self.hidden_dim = hidden_dim
        self.act = ACTIVATIONS[activation]
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
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))
