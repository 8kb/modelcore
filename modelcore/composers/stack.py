import torch
import torch.nn as nn

from modelcore.catalog import register_component
from modelcore.components.contracts import FeatureHost
from modelcore.composers.base import BaseComposer


@register_component("stack")
class StackComposer(BaseComposer, FeatureHost):
    """Sequential residual stack -- the one body class. Threads x0 (the post-embedding
    activations) and a fresh kv_bus dict through every block each forward pass (a producer layer
    writes its K/V into the bus, a consumer layer reads an earlier layer's out of it -- see
    modelcore.components.attention.CausalSelfAttention); a block that doesn't use either is
    unaffected. Stack-level tricks (backout) are features on the hook points below; `state` is a
    dict private to one forward pass, for a feature to remember something between hooks.

    Hook points: `after_block`(x, i, state) -> x, `finish`(x, state) -> x."""
    HOOK_POINTS = ("after_block", "finish")

    def __init__(self, blocks, features):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        self._attach_features(features)

    @torch.no_grad()
    def init_weights(self):
        for block in self.blocks:
            block.init_weights()
        self._init_features()

    def layer_specs(self):
        return [b.layer_spec() for b in self.blocks]

    def forward(self, x, idx, kv_cache, doc_args=None):
        x0 = x
        kv_bus = {}
        state = {}
        for i, block in enumerate(self.blocks):
            x = block(x, x0, idx, kv_cache, kv_bus, doc_args)
            x = self._hook("after_block", x, i, state)
        return self._hook("finish", x, state)
