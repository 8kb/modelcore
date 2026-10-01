"""
The three module contracts every modelcore component obeys: token ids in, residual-stream
activations out (BaseEmbedding); one residual-stream transform step (BaseBlock); residual-stream
activations out, logits or loss (BaseUnembedding). Internal to modelcore -- a component knows it
must implement these, but nothing about how or by whom it's assembled into a model (that's
modelcore.model.Model, driven by a config tree).
"""
import torch.nn as nn


def is_int(v):
    """An int that is not a bool -- what the component validators mean by 'an integer param'."""
    return isinstance(v, int) and not isinstance(v, bool)


class BaseEmbedding(nn.Module):
    """Token ids -> residual-stream activations, ready for the trunk. Owns everything about
    getting from ids to that first activation: the token embedding table, and any input-side
    per-token mixing (e.g. a smear) that needs read/write access to kv_cache.state."""

    def init_weights(self):
        raise NotImplementedError

    def forward(self, idx, kv_cache=None):
        raise NotImplementedError


class BaseFeature(nn.Module):
    """A small, optional trick a host component carries in its `features` list (a gate, a value
    embedding, per-layer lambdas, ...). HOOKS names the hook points this feature implements, each
    as a method of the same name; a host declares the points it exposes in HOOK_POINTS and calls
    every feature at each of them (see FeatureHost). Every hook has the shape
    `hook(value, *context) -> value`, so several features on one point simply chain in list order.
    bind() runs once, from the host's constructor, so a feature can size itself off the host's
    geometry (shapes only -- it may run under torch.device("meta"))."""
    HOOKS = ()

    def bind(self, host):
        pass

    def init_weights(self):
        pass

    def state_elems(self):
        """Elements of per-row inference state this feature keeps in the cache (0 for none)."""
        return 0

    def fwd_flops_per_token(self):
        """Forward FLOPs per token beyond Linear matmuls (0 for none). Shapes only -- both stats
        methods run on a meta-device model."""
        return 0


class FeatureHost:
    """Mixin for a component that carries features. HOOK_POINTS is what the host promises to call;
    validation rejects a feature whose HOOKS are not a subset of it. Features live in a ModuleDict
    keyed by their `#type`, so state_dict keys don't depend on list order
    (`mixer.features.output_gate.proj.weight`)."""
    HOOK_POINTS = ()

    def _attach_features(self, features):
        self.features = nn.ModuleDict()
        for feature in features:
            self.features[feature.COMPONENT_TYPE] = feature
            feature.bind(self)

    def _hook(self, point, value, *context):
        assert point in self.HOOK_POINTS, f"{type(self).__name__} does not expose hook {point!r}"
        for feature in self.features.values():
            if point in feature.HOOKS:
                value = getattr(feature, point)(value, *context)
        return value

    def _init_features(self):
        for feature in self.features.values():
            feature.init_weights()


class BaseMixer(nn.Module):
    """The token-mixing slot of a Block: attention, and later SSMs and convolutions. Owns
    everything about how positions exchange information -- its own state/geometry (layer_spec())
    and any per-layer parameter it introduces. `cache` is the inference cache (None in training),
    `bus` a per-forward dict shared between layers of one stack (KV sharing uses it), `doc_args`
    the intra-document masking data."""

    def init_weights(self):
        raise NotImplementedError

    def bind_layer(self, layer_idx):
        """Called once by the owning Block, so a mixer that needs to know its position (default KV
        slot) doesn't take it as a config param."""

    def forward(self, x, idx, cache, bus=None, doc_args=None):
        raise NotImplementedError

    def layer_spec(self):
        """AttentionLayerSpec (a KV cache), RecurrentLayerSpec (a fixed-size state), or None."""
        return None


class BaseBlock(nn.Module):
    """One residual-stream transform step. Owns everything about its own layer; its attention
    geometry, if any, comes from its mixer via layer_spec()."""

    def init_weights(self):
        raise NotImplementedError

    def forward(self, x, x0, idx, kv_cache, kv_bus=None, doc_args=None):
        raise NotImplementedError

    def layer_spec(self):
        """AttentionLayerSpec, RecurrentLayerSpec, or None -- whatever its mixer reports."""
        return None


class BaseUnembedding(nn.Module):
    """Residual-stream activations -> logits, or loss when targets are given. Owns the final
    norm, the output projection, and the loss computation."""

    def init_weights(self):
        raise NotImplementedError

    def forward(self, x, targets=None, loss_reduction="mean"):
        raise NotImplementedError
