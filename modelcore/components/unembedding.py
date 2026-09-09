import torch
import torch.nn as nn
import torch.nn.functional as F

from modelcore.catalog import register_component
from modelcore.components.contracts import BaseUnembedding
from modelcore.components.linear import Linear
from modelcore.components.norm import norm


@register_component("lm_head", needs=("n_embd", "vocab_size", "padded_vocab_size"))
class LMHead(BaseUnembedding):
    """Final norm() + output projection + vocab crop + tanh softcap, and the loss when targets
    are given. Residual-stream activations -> logits.

    weight= optionally ties the projection to an existing Parameter (e.g. an embedding's wte
    weight) instead of allocating and training its own: when tied, this module declares no role
    for that parameter via the param_roles() escape hatch, since whoever passed it in already
    owns (and declared a role for) it -- see modelcore.roles."""

    def __init__(self, n_embd, vocab_size, padded_vocab_size, softcap=15, weight=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.softcap = softcap
        self.lm_head = Linear(n_embd, padded_vocab_size, bias=False)
        self._tied = weight is not None
        if self._tied:
            self.lm_head.weight = weight

    def param_roles(self):
        if self._tied:
            return {}
        return {"unembedding": [self.lm_head.weight]}

    @torch.no_grad()
    def init_weights(self):
        if not self._tied:
            torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        # if tied, real values are set wherever the owning module's init_weights() runs

    def forward(self, x, targets=None, loss_reduction="mean"):
        x = norm(x)
        logits = self.lm_head(x)  # (B, T, padded_vocab_size) <- very big tensor, large amount of memory
        logits = logits[..., :self.vocab_size]  # slice to remove padding
        logits = logits.float()  # switch to fp32 for logit softcap and loss computation
        logits = self.softcap * torch.tanh(logits / self.softcap)  # smoothly cap logits to [-softcap, softcap]
        if targets is not None:
            # training: given the targets, compute and return the loss
            return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
        # inference: just return the logits directly
        return logits
