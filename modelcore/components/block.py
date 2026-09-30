import torch

from modelcore.catalog import get_component, register_component
from modelcore.components.contracts import BaseBlock, BaseMixer, FeatureHost
from modelcore.config.spec import ComponentSpec


def _validate_block(params, ctx):
    mixer = params.get("mixer")
    if isinstance(mixer, ComponentSpec):
        try:
            cls = get_component(mixer.type)[0]
        except ValueError:
            return []  # unknown type: the structural pass already reports it
        if not issubclass(cls, BaseMixer):
            return [f"mixer must be a mixer component, {mixer.type!r} is not one"]
    return []


@register_component("block", needs=("norm",), validate=_validate_block)
class Block(BaseBlock, FeatureHost):
    """The one block class: a pre-norm residual step, x = x + mixer(norm(x)); x = x + ffn(norm(x)).
    Everything that used to distinguish one block type from another is a component in a slot:
    the token mixer (attention today; SSMs and convolutions as they arrive) and the FFN are nested
    specs, and every per-layer trick (the resid/x0 lambdas, ...) is a feature in `features`. So a
    "gpt" block and a "llama" block are the same class with different feature sets. layer_idx is
    handed to the mixer (see BaseMixer.bind_layer); the norm is the config's shared one
    (`shared.norm`).

    Hook points: `residual_in`(x, x0) -> x, applied to the block input before anything else. x0 is
    the stack's post-embedding activations, threaded to every block whether or not a feature reads
    it."""
    HOOK_POINTS = ("residual_in",)

    def __init__(self, layer_idx, mixer, ffn, features, norm):
        super().__init__()
        self.layer_idx = layer_idx
        self.mixer = mixer
        self.mixer.bind_layer(layer_idx)
        self.ffn = ffn
        self.norm = norm
        self._attach_features(features)

    @torch.no_grad()
    def init_weights(self):
        self.mixer.init_weights()
        self.ffn.init_weights()
        self._init_features()

    def layer_spec(self):
        return self.mixer.layer_spec()

    def forward(self, x, x0, idx, kv_cache, kv_bus=None, doc_args=None):
        x = self._hook("residual_in", x, x0)
        x = x + self.mixer(self.norm(x), idx, kv_cache, kv_bus, doc_args)
        x = x + self.ffn(self.norm(x))
        return x
