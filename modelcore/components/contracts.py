"""
The three module contracts every modelcore component obeys: token ids in, residual-stream
activations out (BaseEmbedding); one residual-stream transform step (BaseBlock); residual-stream
activations out, logits or loss (BaseUnembedding). Internal to modelcore -- a component knows it
must implement these, but nothing about how or by whom it's assembled into a model (that's
modelcore.model.Model, driven by a config tree).
"""
import torch.nn as nn


class BaseEmbedding(nn.Module):
    """Token ids -> residual-stream activations, ready for the trunk. Owns everything about
    getting from ids to that first activation: the token embedding table, and any input-side
    per-token mixing (e.g. a smear) that needs read/write access to kv_cache.state."""

    def init_weights(self):
        raise NotImplementedError

    def forward(self, idx, kv_cache=None):
        raise NotImplementedError


class BaseBlock(nn.Module):
    """One residual-stream transform step. Owns everything about its own layer: attention
    geometry (via layer_spec()) and any per-layer parameter this block introduces."""

    def init_weights(self):
        raise NotImplementedError

    def forward(self, x, x0, idx, kv_cache, kv_bus=None, doc_args=None):
        raise NotImplementedError

    def layer_spec(self):
        """AttentionLayerSpec for this layer, or None if this block isn't attention-shaped."""
        return None


class BaseUnembedding(nn.Module):
    """Residual-stream activations -> logits, or loss when targets are given. Owns the final
    norm, the output projection, and the loss computation."""

    def init_weights(self):
        raise NotImplementedError

    def forward(self, x, targets=None, loss_reduction="mean"):
        raise NotImplementedError
