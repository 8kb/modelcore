"""
The feature protocol: a feature declares HOOKS, a host declares HOOK_POINTS and calls every feature
at each, validation rejects a feature on a host without its hook and a duplicate type, and features
live in a ModuleDict keyed by #type so state_dict keys don't depend on list order.

python -m pytest modelcore/tests/test_features.py -v
"""
import pytest
import torch

from modelcore import ComponentSpec, ModelManager
from modelcore.catalog import register_component
from modelcore.components.contracts import BaseFeature
from modelcore.tests.conftest import FLAVORS, _gate, _nanogpt_like, _plain_like, build


@register_component("test_scale")
class _Scale(BaseFeature):
    """Test-only feature: multiplies the attention output by a constant."""
    HOOKS = ("output",)

    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def output(self, y, x):
        return y * self.factor


@register_component("test_scale_two")
class _ScaleTwo(_Scale):
    pass


def _errors(manager, config):
    return [f"{e.path}: {e.message}" for e in manager.validate_config(config).errors]


def _mixer(config, i=0):
    return config.body.params["blocks"][i].params["mixer"]


def test_a_feature_on_a_host_without_its_hook_is_rejected(manager):
    config = _plain_like()
    config.body.params["features"] = [ComponentSpec("output_gate", {"granularity": "head", "block_size": None,
                                                                    "in_channels": None})]
    errors = _errors(manager, config)
    assert any("does not expose hook" in e and "'output'" in e for e in errors), errors

    config = _plain_like()
    config.body.params["blocks"][0].params["features"] = [ComponentSpec("backout", {"backout_layer": 0,
                                                                                      "backout_lambda_init": 0.2})]
    assert any("does not expose hook" in e for e in _errors(manager, config))


def test_a_duplicate_feature_type_is_rejected(manager):
    config = _plain_like(attn_gate=_gate("head"))
    _mixer(config).params["features"].append(_gate("element")())
    assert any("duplicate feature 'output_gate'" in e for e in _errors(manager, config))


def test_something_that_is_not_a_feature_is_rejected(manager):
    config = _plain_like()
    _mixer(config).params["features"] = [ComponentSpec("mlp", {"activation": "relu", "hidden_dim": 8})]
    assert any("is not a feature" in e for e in _errors(manager, config))


def test_features_must_be_a_list(manager):
    config = _plain_like()
    _mixer(config).params["features"] = "output_gate"
    assert any("must be a list" in e for e in _errors(manager, config))


def test_a_block_mixer_must_be_a_mixer(manager):
    config = _plain_like()
    config.body.params["blocks"][0].params["mixer"] = ComponentSpec("mlp", {"activation": "relu", "hidden_dim": 8})
    assert any("must be a mixer component" in e for e in _errors(manager, config))


def test_value_embed_on_a_kv_sharing_consumer_is_rejected(manager):
    config = _plain_like(kv_slots=[0, 1, 1, 1])
    _mixer(config, 2).params["features"] = [ComponentSpec("value_embed", {"gate_channels": 12})]
    assert any("cannot have a value_embed" in e for e in _errors(manager, config))


def test_unknown_feature_type_is_reported(manager):
    config = _plain_like()
    _mixer(config).params["features"] = [ComponentSpec("nonexistent_trick", {})]
    assert any("unknown component type" in e for e in _errors(manager, config))


def test_state_dict_keys_do_not_depend_on_feature_list_order(manager):
    a = _nanogpt_like(attn_gate=_gate("head"))
    b = _nanogpt_like(attn_gate=_gate("head"))
    for block in b.body.params["blocks"]:
        block.params["mixer"].params["features"].reverse()
    keys_a, keys_b = set(build(manager, a).state_dict()), set(build(manager, b).state_dict())
    assert keys_a == keys_b
    assert "body.blocks.0.mixer.features.output_gate.proj.weight" in keys_a
    assert "body.blocks.0.features.resid_lambdas.resid_lambda" in keys_a
    assert "body.features.backout.backout_lambda" in keys_a
    assert any(k.startswith("body.blocks.3.mixer.features.value_embed.embed") for k in keys_a)


def test_a_new_feature_is_a_registered_type_not_a_format_change(manager):
    """A trick nothing in modelcore knows about beyond its registration rides the same list."""
    config = _plain_like()
    _mixer(config).params["features"] = [ComponentSpec("test_scale", {"factor": 3.0})]
    assert manager.validate_config(config).ok
    model = build(manager, config)
    assert list(model.body.blocks[0].mixer.features) == ["test_scale"]


def test_features_chain_at_a_hook_point_in_list_order(manager):
    def out(features):
        config = _plain_like(n_layer=1)
        _mixer(config).params["features"] = features
        model = build(manager, config, seed=1)
        with torch.no_grad():
            model.body.blocks[0].mixer.c_proj.weight.normal_()
            model.body.blocks[0].ffn.down_proj.weight.zero_()  # isolate the mixer's contribution
        idx = torch.randint(0, 64, (1, 6))
        return model, idx

    plain_model, idx = out([])
    scaled_model, _ = out([ComponentSpec("test_scale", {"factor": 2.0}), ComponentSpec("test_scale_two", {"factor": 5.0})])
    with torch.no_grad():
        scaled_model.load_state_dict(plain_model.state_dict())
        x = plain_model.embedding(idx)
        mixer_in = plain_model.body.blocks[0].norm(x)
        y_plain = plain_model.body.blocks[0].mixer(mixer_in, idx, None)
        y_scaled = scaled_model.body.blocks[0].mixer(mixer_in, idx, None)
    assert torch.allclose(y_scaled, 10.0 * y_plain, rtol=1e-4, atol=1e-4)  # the scale lands before c_proj's matmul


def test_hook_call_outside_the_declared_points_is_an_error(manager):
    model = build(manager, _plain_like())
    with pytest.raises(AssertionError, match="does not expose hook"):
        model.body.blocks[0]._hook("output", None)
