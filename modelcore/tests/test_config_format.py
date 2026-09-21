"""
The config *format*: modelcore.v2 serialization, `_` comments, the v1 -> v2 upgrade, and the rule
that a v2 tree has no defaults. Before this file nothing in modelcore's own suite exercised
to_dict/from_dict at all -- the only thing pinning it was a host's checkpoint round-trip test.

python -m pytest modelcore/tests/test_config_format.py -v
"""
import copy
import json

import pytest
import torch

from modelcore import AdapterSpec, ComponentSpec, ModelConfig
from modelcore.config.spec import FORMAT, FORMAT_V1, SUPPORTED_FORMATS, TEMPLATES
from modelcore.config.upgrade import upgrade_v1_to_v2
from modelcore.tests.conftest import FLAVORS, build


def _json_roundtrip(d):
    return json.loads(json.dumps(d))


# -----------------------------------------------------------------------------
# v2 round trip

def test_v2_round_trip_is_lossless_for_every_flavor(config):
    d = config.to_dict()
    assert d["format"] == FORMAT == "modelcore.v2"
    again = ModelConfig.from_dict(_json_roundtrip(d))
    assert again.to_dict() == d
    assert again == config


def test_optional_extensions_are_omitted_when_empty_and_kept_when_set(config):
    d = config.to_dict()
    assert "meta" not in d and "tokenizer" not in d
    config.meta = {"name": "tiny", "created": "2026-09-22"}
    config.tokenizer = {"name": "bpe32k", "fingerprint": "abc123", "vocab_size": 128}
    again = ModelConfig.from_dict(_json_roundtrip(config.to_dict()))
    assert again.meta == config.meta and again.tokenizer == config.tokenizer


def test_meta_and_tokenizer_are_carried_never_interpreted(manager, config):
    """modelcore knows nothing about tokenizers: a descriptor whose vocab_size disagrees with the
    config's own is still just data here (a host reconciles it)."""
    config.tokenizer = {"vocab_size": config.vocab_size + 999, "anything": ["goes"]}
    config.meta = {"nested": {"x": 1}}
    assert manager.validate_config(config).ok
    build(manager, config)


# -----------------------------------------------------------------------------
# comments

def _commented():
    config = FLAVORS["gpt_lora"]()
    config.comments = {"_comment": "top level", "_why": "a test"}
    config.shared["rope"].comments = {"_note": "on a shared component"}
    config.input.comments = {"_note": "on the embedding"}
    block = config.body.params["blocks"][0]
    block.comments = {"_comment": "on a block"}
    block.params["mlp"].comments = {"_comment": "on the nested mlp"}
    config.adapters[0].comments = {"_note": "on an adapter"}
    config.adapters[0].params_comments = {"_note": "inside adapter params"}
    return config


def test_comments_round_trip_at_every_level():
    config = _commented()
    d = _json_roundtrip(config.to_dict())
    assert d["_comment"] == "top level"
    assert d["shared"]["rope"]["_note"] == "on a shared component"
    assert d["body"]["blocks"][0]["_comment"] == "on a block"
    assert d["body"]["blocks"][0]["mlp"]["_comment"] == "on the nested mlp"
    assert d["adapters"][0]["_note"] == "on an adapter"
    assert d["adapters"][0]["params"]["_note"] == "inside adapter params"
    again = ModelConfig.from_dict(d)
    assert again == config
    assert again.to_dict() == config.to_dict()


def test_a_comment_can_never_reach_a_constructor_or_fail_validation(manager):
    """The safety property: comments live beside params, never in them, so they aren't
    constructor kwargs (which validation would flag as "no such param") and don't stop a build."""
    config = ModelConfig.from_dict(_json_roundtrip(_commented().to_dict()))
    block = config.body.params["blocks"][0]
    assert "_comment" not in block.params and "_comment" not in block.params["mlp"].params
    assert "_note" not in config.adapters[0].params
    assert manager.validate_config(config).ok
    build(manager, config)


def test_an_unknown_non_comment_key_in_a_spec_is_still_an_error(manager):
    """...whereas a *mistyped parameter* stays an error -- that distinction is what the `_` prefix
    buys."""
    config = FLAVORS["gpt"]()
    config.body.params["blocks"][0].params["comment"] = "missing its underscore"
    report = manager.validate_config(config)
    assert not report.ok
    assert any("no such param" in e.message and e.path.endswith(".comment") for e in report.errors)


# -----------------------------------------------------------------------------
# format dispatch

def test_unknown_format_is_rejected_naming_the_supported_ones(config):
    d = config.to_dict()
    d["format"] = "modelcore.v9"
    with pytest.raises(ValueError, match="unsupported config format") as e:
        ModelConfig.from_dict(d)
    for supported in SUPPORTED_FORMATS:
        assert supported in str(e.value)


def test_unknown_top_level_key_is_an_error_not_silently_dropped(config):
    d = config.to_dict()
    d["adaptrs"] = []  # a typo -- used to vanish, and then be lost on the next save
    with pytest.raises(ValueError, match="unknown top-level key"):
        ModelConfig.from_dict(d)


def test_underscore_top_level_key_is_a_comment_not_an_error(config):
    d = config.to_dict()
    d["_todo"] = "fine"
    assert ModelConfig.from_dict(d).comments == {"_todo": "fine"}


@pytest.mark.parametrize("missing", ["template", "pad_vocab_size_to", "sequence_len", "vocab_size", "n_embd",
                                     "input", "body", "output"])
def test_v2_dict_missing_a_required_key_is_rejected(config, missing):
    d = config.to_dict()
    del d[missing]
    with pytest.raises(ValueError, match="v2 has no defaults") as e:
        ModelConfig.from_dict(d)
    assert missing in str(e.value)


def test_v2_adapter_must_state_enabled():
    d = FLAVORS["gpt_lora"]().to_dict()
    del d["adapters"][0]["enabled"]
    with pytest.raises(KeyError):
        ModelConfig.from_dict(d)


# -----------------------------------------------------------------------------
# a v2 tree has no defaults: what a config omits, validation rejects

def _validation_messages(manager, config):
    report = manager.validate_config(config)
    return report, " | ".join(f"{e.path}: {e.message}" for e in report.errors)


def test_block_without_mlp_is_rejected(manager):
    for flavor in ("gpt", "llama"):
        config = FLAVORS[flavor]()
        del config.body.params["blocks"][0].params["mlp"]
        report, messages = _validation_messages(manager, config)
        assert not report.ok and "missing required param 'mlp'" in messages
        with pytest.raises(ValueError):
            manager.create_model(config, device=torch.device("cpu"))


def test_config_without_shared_norm_is_rejected(manager):
    config = FLAVORS["gpt"]()
    del config.shared["norm"]
    report, messages = _validation_messages(manager, config)
    assert not report.ok and "needs 'norm'" in messages


def _target(config, where):
    """The ComponentSpec a (flavor-relative) location names."""
    if where == "output":
        return config.output
    if where == "input":
        return config.input
    if where == "body":
        return config.body
    if where == "rope":
        return config.shared["rope"]
    if where == "block0":
        return config.body.params["blocks"][0]
    raise AssertionError(where)


@pytest.mark.parametrize("flavor,where,param", [
    ("gpt", "output", "softcap"),
    ("gpt", "input", "smear"),
    ("gpt", "rope", "over_compute"),
    ("gpt", "body", "backout_lambda_init"),
    ("llama", "block0", "kv_slot"),
    ("llama", "block0", "produces_kv"),
])
def test_formerly_defaulted_params_are_now_required(manager, flavor, where, param):
    config = FLAVORS[flavor]()
    del _target(config, where).params[param]
    report, messages = _validation_messages(manager, config)
    assert not report.ok and f"missing required param '{param}'" in messages


def test_template_must_be_a_supported_one(manager):
    config = FLAVORS["gpt"]()
    for ok in TEMPLATES:
        config.template = ok
        assert manager.validate_config(config).ok
    config.template = "chatml"
    report, messages = _validation_messages(manager, config)
    assert not report.ok and "template" in messages and "chatml" in messages


def test_mlp_spec_is_validated(manager):
    config = FLAVORS["gpt"]()
    mlp = config.body.params["blocks"][0].params["mlp"]
    mlp.params["activation"] = "swish"
    mlp.params["hidden_dim"] = 0
    report, messages = _validation_messages(manager, config)
    assert not report.ok and "activation must be one of" in messages and "hidden_dim must be a positive" in messages
    # a gated mlp does not take relu2 (that is a plain-mlp activation)
    config = FLAVORS["llama"]()
    config.body.params["blocks"][0].params["mlp"].params["activation"] = "relu2"
    report, _ = _validation_messages(manager, config)
    assert not report.ok


def test_layer_norm_eps_must_be_a_number_but_rms_norm_may_be_null(manager):
    config = FLAVORS["gpt"]()
    assert config.shared["norm"].params == {"eps": None}
    assert manager.validate_config(config).ok
    config.shared["norm"] = ComponentSpec("layer_norm", {"eps": None})
    assert not manager.validate_config(config).ok


# -----------------------------------------------------------------------------
# the mlp / norm choices actually reach the model

def test_nested_mlp_spec_params_reach_the_module(manager):
    config = FLAVORS["gpt_gated_mlp"]()
    model = build(manager, config)
    mlp = model.body.blocks[0].mlp
    assert type(mlp).__name__ == "GatedMLP" and mlp.hidden_dim == 100
    assert mlp.gate_proj.weight.shape == (100, config.n_embd)


def test_different_ffn_choices_give_different_models(manager):
    a = build(manager, FLAVORS["gpt"]())
    b = build(manager, FLAVORS["gpt_gated_mlp"]())
    assert sum(p.numel() for p in a.parameters()) != sum(p.numel() for p in b.parameters())


def test_a_configurable_norm_adds_nothing_to_the_state_dict(manager, config):
    """A shared, parameterless norm must leave checkpoints unchanged -- that is what lets every
    pre-existing checkpoint load. No state_dict key may mention it."""
    model = build(manager, config)
    assert not any("norm" in k for k in model.state_dict())
    assert "shared.norm" not in " ".join(model.state_dict())


# -----------------------------------------------------------------------------
# v1 -> v2

# What v1 left implicit -- the converter's own defaults, mirrored here so a v2 dict can be
# "downgraded" to a v1-shaped one and the round trip checked.
_V1_DEFAULTS = {"softcap": 15, "smear": True, "over_compute": 10, "backout_lambda_init": 0.2,
                "kv_slot": None, "produces_kv": True}


def _downgrade(node, n_embd, sequence_len):
    """v2 dict -> what the v1 code would have written: drop whatever equals a v1 default."""
    if isinstance(node, list):
        return [_downgrade(v, n_embd, sequence_len) for v in node]
    if not isinstance(node, dict):
        return node
    out = {}
    for k, v in node.items():
        if node.get("#type") in ("gpt_block", "plain_block") and k == "mlp":
            default = upgrade_v1_to_v2({"n_embd": n_embd, "sequence_len": sequence_len,
                                        "body": {"#type": node["#type"]}})["body"]["mlp"]
            if v == default:
                continue
        if k in _V1_DEFAULTS and v == _V1_DEFAULTS[k] and "#type" in node:
            continue
        out[k] = _downgrade(v, n_embd, sequence_len)
    return out


def _as_v1(d):
    d = copy.deepcopy(d)
    n_embd, seq = d["n_embd"], d["sequence_len"]
    d["format"] = FORMAT_V1
    if d.pop("template") != "base":
        pytest.skip("only 'base' is implicit in v1")
    if d["pad_vocab_size_to"] == 64:
        del d["pad_vocab_size_to"]
    if d["shared"].get("norm") == {"#type": "rms_norm", "eps": None}:
        del d["shared"]["norm"]
    for section in ("shared", "input", "body", "output"):
        d[section] = _downgrade(d[section], n_embd, seq)
    for a in d.get("adapters", []):
        if a["enabled"] is True:
            del a["enabled"]
    return d


def test_v1_dict_upgrades_to_exactly_the_v2_dict_for_every_flavor(config):
    v2 = config.to_dict()
    v1 = _as_v1(v2)
    assert v1["format"] == FORMAT_V1
    assert ModelConfig.from_dict(_json_roundtrip(v1)).to_dict() == v2


def test_a_dict_with_no_format_key_is_treated_as_v1(config):
    v1 = _as_v1(config.to_dict())
    del v1["format"]
    assert ModelConfig.from_dict(v1).to_dict() == config.to_dict()


def test_upgrade_does_not_mutate_its_input(config):
    v1 = _as_v1(config.to_dict())
    before = copy.deepcopy(v1)
    upgrade_v1_to_v2(v1)
    assert v1 == before


def test_upgrade_materializes_what_v1_hardcoded():
    d = upgrade_v1_to_v2({"format": FORMAT_V1, "sequence_len": 32, "vocab_size": 128, "n_embd": 768,
                          "shared": {},
                          "input": {"#type": "token_embedding"}, "output": {"#type": "lm_head"},
                          "body": {"#type": "stack", "blocks": [
                              {"#type": "gpt_block", "window": 32},
                              {"#type": "plain_block", "window": 8}]}})
    gpt, plain = d["body"]["blocks"]
    assert d["template"] == "base" and d["pad_vocab_size_to"] == 64
    assert d["shared"]["norm"] == {"#type": "rms_norm", "eps": None}
    assert gpt["mlp"] == {"#type": "mlp", "activation": "relu2", "hidden_dim": 4 * 768}
    # int(2 * 4 * 768 / 3) = 2048, already a multiple of 256
    assert plain["mlp"] == {"#type": "gated_mlp", "activation": "silu", "hidden_dim": 2048}
    assert (plain["kv_slot"], plain["produces_kv"]) == (None, True)
    assert d["input"]["smear"] is True and d["output"]["softcap"] == 15


def test_upgrade_rounds_the_gated_width_up_to_a_multiple_of_256():
    d = upgrade_v1_to_v2({"format": FORMAT_V1, "sequence_len": 32, "vocab_size": 128, "n_embd": 64, "shared": {},
                          "input": {"#type": "token_embedding"}, "output": {"#type": "lm_head"},
                          "body": {"#type": "stack", "blocks": [{"#type": "plain_block", "window": -1}]}})
    assert d["body"]["blocks"][0]["mlp"]["hidden_dim"] == 256  # int(2*256/3)=170 -> 256


def test_upgrade_never_overrides_a_value_the_dict_already_states():
    d = upgrade_v1_to_v2({"format": FORMAT_V1, "sequence_len": 32, "vocab_size": 128, "n_embd": 64,
                          "template": "nanochat", "pad_vocab_size_to": 128,
                          "shared": {"norm": {"#type": "layer_norm", "eps": 1e-6}},
                          "input": {"#type": "token_embedding", "smear": False},
                          "output": {"#type": "lm_head", "softcap": 30},
                          "body": {"#type": "stack", "blocks": [
                              {"#type": "gpt_block", "window": -1, "mlp": {"#type": "mlp", "activation": "gelu", "hidden_dim": 7}}]}})
    assert d["template"] == "nanochat" and d["pad_vocab_size_to"] == 128
    assert d["shared"]["norm"] == {"#type": "layer_norm", "eps": 1e-6}
    assert d["input"]["smear"] is False and d["output"]["softcap"] == 30
    assert d["body"]["blocks"][0]["mlp"] == {"#type": "mlp", "activation": "gelu", "hidden_dim": 7}


def test_upgrade_normalizes_full_context_windows_to_minus_one():
    """window == sequence_len was how v1 spelled 'full attention'; it is -1 now. A real, shorter
    window is left alone."""
    blocks = [{"#type": "plain_block", "window": w} for w in (2048, 4096, 512, -1)]
    d = upgrade_v1_to_v2({"format": FORMAT_V1, "sequence_len": 2048, "vocab_size": 128, "n_embd": 64, "shared": {},
                          "input": {"#type": "token_embedding"}, "output": {"#type": "lm_head"},
                          "body": {"#type": "stack", "blocks": blocks}})
    assert [b["window"] for b in d["body"]["blocks"]] == [-1, -1, 512, -1]


def test_upgrade_walks_nested_composers():
    inner = {"#type": "stack", "blocks": [{"#type": "gpt_block", "window": 32}]}
    d = upgrade_v1_to_v2({"format": FORMAT_V1, "sequence_len": 32, "vocab_size": 128, "n_embd": 64, "shared": {},
                          "input": {"#type": "token_embedding"}, "output": {"#type": "lm_head"},
                          "body": {"#type": "backout", "backout_layer": 0, "blocks": [], "inner": inner}})
    assert d["body"]["backout_lambda_init"] == 0.2
    nested = d["body"]["inner"]["blocks"][0]
    assert nested["window"] == -1 and nested["mlp"]["hidden_dim"] == 256


def test_a_v1_config_still_builds_the_same_model_as_the_v2_one(manager):
    """The converter's materialized values must reproduce what v1 hardcoded: same parameters,
    same shapes, and -- given the same weights -- the same logits."""
    for name in ("gpt", "llama", "llama_kvshare", "llama_kvshare_win"):
        v2 = FLAVORS[name]()
        upgraded = ModelConfig.from_dict(_as_v1(v2.to_dict()))
        a, b = build(manager, v2), build(manager, upgraded)
        assert {k: v.shape for k, v in a.state_dict().items()} == {k: v.shape for k, v in b.state_dict().items()}
        b.load_state_dict(a.state_dict())
        idx = torch.randint(0, v2.vocab_size, (2, 8))
        with torch.no_grad():
            assert torch.equal(a(idx), b(idx)), name
