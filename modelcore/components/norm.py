"""
Normalization as a config-selectable component. A config's `shared.norm` names one instance that
every block/embedding/unembedding needing a norm (needs=(..., "norm")) shares -- including
attention's QK-norm, which the block passes down.

Both are parameterless on purpose: no learnable gain. A gain would need PARAM_ROLES and would
change the optimizer's positional on-disk layout (see AGENTS.md), and a parameterless module adds
nothing to state_dict(), so a shared instance costs a checkpoint nothing.
"""
import torch.nn as nn
import torch.nn.functional as F

from modelcore.catalog import register_component


def _validate_rms_norm(params, ctx):
    eps = params.get("eps")
    if eps is not None and not (isinstance(eps, (int, float)) and eps > 0):
        return [f"eps must be null (the input dtype's machine epsilon) or a positive number, got {eps!r}"]
    return []


def _validate_layer_norm(params, ctx):
    eps = params.get("eps")
    if not (isinstance(eps, (int, float)) and eps > 0):
        return [f"eps must be a positive number, got {eps!r}"]
    return []


@register_component("rms_norm", validate=_validate_rms_norm)
class RMSNorm(nn.Module):
    """F.rms_norm over the last dim. eps=None is torch's own default -- the input dtype's machine
    epsilon -- which is what every model trained before norm was configurable used, so it is a real
    value here rather than a placeholder."""

    def __init__(self, eps):
        super().__init__()
        self.eps = eps

    def init_weights(self):
        pass  # nothing to initialize; Model.init_weights calls this on every shared component

    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)  # note that this will run in bf16, seems ok


@register_component("layer_norm", validate=_validate_layer_norm)
class LayerNorm(nn.Module):
    """F.layer_norm over the last dim, no gain/bias."""

    def __init__(self, eps):
        super().__init__()
        self.eps = eps

    def init_weights(self):
        pass

    def forward(self, x):
        return F.layer_norm(x, (x.size(-1),), eps=self.eps)
