import torch.nn as nn


class BaseComposer(nn.Module):
    """Contract: forward(x, idx, kv_cache) -> x; layer_specs() -> list[AttentionLayerSpec], in
    forward-pass order (a composer wrapping several block lists concatenates them so the generic
    accounting in modelcore.stats keeps working unchanged). A composer owns a set of blocks (or
    other composers) and how they connect into one residual-stream transform -- a nameable,
    swappable thing a config's `body` names via its own "#type"."""

    def init_weights(self):
        raise NotImplementedError

    def forward(self, x, idx, kv_cache, doc_args=None):
        raise NotImplementedError

    def layer_specs(self):
        raise NotImplementedError
