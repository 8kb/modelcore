import torch
import torch.nn as nn

from modelcore.catalog import register_component
from modelcore.composers.base import BaseComposer


@register_component("backout")
class BackoutComposer(BaseComposer):
    """x0 (post-embedding activations) blended into every block via each block's own resid/x0
    lambdas, plus a mid-depth "backout" subtraction before the final norm. Owns backout_lambda
    itself (SOLID module ownership: the composer that introduces this residual topology owns the
    one scalar parameter it needs, rather than it dangling on some unrelated top-level object)."""
    PARAM_ROLES = {"backout_lambda": "backout_scalar"}

    def __init__(self, blocks, backout_layer, backout_lambda_init=0.2):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        self.backout_layer = backout_layer
        self.backout_lambda = nn.Parameter(torch.empty(()))  # fake init, real init in init_weights()
        self._backout_lambda_init = backout_lambda_init

    @torch.no_grad()
    def init_weights(self):
        for block in self.blocks:
            block.init_weights()
        self.backout_lambda.fill_(self._backout_lambda_init)

    def layer_specs(self):
        return [b.layer_spec() for b in self.blocks]

    def forward(self, x, idx, kv_cache, doc_args=None):
        x0 = x  # save initial (post-embedding) activations for the x0 residual
        x_backout = None
        for i, block in enumerate(self.blocks):
            x = block(x, x0, idx, kv_cache, doc_args=doc_args)
            if i == self.backout_layer:
                x_backout = x
        if x_backout is not None:
            x = x - self.backout_lambda.to(x.dtype) * x_backout
        return x
