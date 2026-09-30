import torch
import torch.nn as nn

from modelcore.catalog import register_component
from modelcore.components.linear import Linear

GRANULARITIES = ("head", "element", "block")


def _is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def _validate_attn_gate(params, ctx):
    errors = []
    granularity = params.get("granularity")
    block_size = params.get("block_size")
    in_channels = params.get("in_channels")
    n_embd = ctx.get("n_embd")
    if granularity not in GRANULARITIES:
        errors.append(f"granularity must be one of {list(GRANULARITIES)}, got {granularity!r}")
    elif granularity == "block":
        if not _is_int(block_size) or block_size <= 0:
            errors.append(f"block_size must be a positive integer for granularity 'block', got {block_size!r}")
    elif block_size is not None:
        errors.append(f"block_size must be null for granularity {granularity!r}, got {block_size!r}")
    if in_channels is not None:
        if not _is_int(in_channels) or in_channels < 1 or (n_embd is not None and in_channels > n_embd):
            errors.append(f"in_channels must be null or an integer in [1, n_embd={n_embd}], got {in_channels!r}")
    return errors


@register_component("attn_gate", needs=("n_embd",), validate=_validate_attn_gate)
class AttnGate(nn.Module):
    """Output gate of attention (Qwen "Gated Attention", G1 placement): after SDPA and before
    c_proj, y = y * sigmoid(W_g . x[..., :in_channels]) where x is the block's normed attention
    input. granularity picks how many gates each head gets: `head` one, `element` one per output
    channel (head_dim), `block` one per contiguous block_size outputs (head_dim // block_size).
    in_channels=None means all n_embd residual channels feed the gate.

    The projection is built in bind(), not __init__: a nested spec is built before its parent, so
    the gate cannot know n_head/head_dim yet. CausalSelfAttention calls bind() from its own
    constructor (shapes only, so still meta-device-safe). Being a Linear, the projection gets the
    default "matrix" role and is counted by stats.num_matmul_params."""

    def __init__(self, n_embd, granularity, block_size, in_channels):
        super().__init__()
        self.granularity = granularity
        self.block_size = block_size
        self.in_channels = n_embd if in_channels is None else in_channels
        self.proj = None
        self.gates_per_head = None

    def bind(self, n_head, head_dim):
        if self.granularity == "head":
            g = 1
        elif self.granularity == "element":
            g = head_dim
        else:
            assert head_dim % self.block_size == 0, f"head_dim ({head_dim}) must be divisible by block_size ({self.block_size})"
            g = head_dim // self.block_size
        self.gates_per_head = g
        self.proj = Linear(self.in_channels, n_head * g, bias=False)

    @torch.no_grad()
    def init_weights(self):
        s = 3**0.5 * self.in_channels**-0.5  # same scheme as c_q
        torch.nn.init.uniform_(self.proj.weight, -s, s)

    def forward(self, x, y):
        B, T, H, D = y.shape
        G = self.gates_per_head
        g = torch.sigmoid(self.proj(x[..., :self.in_channels])).view(B, T, H, G, 1)
        return (y.reshape(B, T, H, G, D // G) * g).view(B, T, H, D)
