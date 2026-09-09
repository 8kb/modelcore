import torch.nn as nn

from modelcore.catalog import register_component
from modelcore.composers.base import BaseComposer


@register_component("stack")
class StackComposer(BaseComposer):
    """Plain sequential residual stack, no x0 residual. Threads a fresh kv_bus dict through every
    block each forward pass (a producer layer writes its K/V into it, a consumer layer reads an
    earlier layer's out of it -- see modelcore.components.attention.CausalSelfAttention); a block
    that ignores kv_bus (BaseBlock.forward's kv_bus=None default) is unaffected."""

    def __init__(self, blocks):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)

    def init_weights(self):
        for block in self.blocks:
            block.init_weights()

    def layer_specs(self):
        return [b.layer_spec() for b in self.blocks]

    def forward(self, x, idx, kv_cache, doc_args=None):
        kv_bus = {}
        for block in self.blocks:
            x = block(x, None, idx, kv_cache, kv_bus, doc_args)
        return x
