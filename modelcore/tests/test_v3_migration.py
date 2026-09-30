"""
Everything about getting from v2 to v3: the config upgrader, the checkpoint converter, and the
guarantee that a v2 checkpoint gives bit-identical logits after both. The goldens in tests/goldens
were written by the last v2 code (awake random weights, a fixed input, the logits it produced), so
"identical" is against real v2 behavior, not against a reimplementation.

python -m pytest modelcore/tests/test_v3_migration.py -v
"""
import glob
import os

import pytest
import torch

from modelcore import ComponentSpec, ModelConfig, ModelManager
from modelcore.config.spec import FORMAT, FORMAT_V2
from modelcore.config.upgrade import remap_v2_name, upgrade_to_v3, upgrade_v2_to_v3
from modelcore.convert import convert_checkpoint_v2_to_v3, convert_state_v2_to_v3, main as convert_main
from modelcore.store import FileSystemStore
from modelcore.tests.conftest import FLAVORS, build
from modelcore.tests.v2_shape import downgrade_v3_to_v2

GOLDEN_DIR = os.path.join(os.path.dirname(__file__), "goldens")
GOLDENS = sorted(glob.glob(os.path.join(GOLDEN_DIR, "v2_*.pt")))


def _load_golden(path):
    return torch.load(path, map_location="cpu")


def _v2_store(tmp_path, golden):
    store = FileSystemStore(str(tmp_path / "v2"), step=0)
    store.write_config(golden["config"])
    store.write_model_state(golden["state"])
    return store


def test_goldens_exist():
    assert len(GOLDENS) >= 5


@pytest.mark.parametrize("path", GOLDENS, ids=os.path.basename)
def test_converted_v2_checkpoint_gives_bit_identical_logits(manager, tmp_path, path):
    golden = _load_golden(path)
    assert golden["config"]["format"] == FORMAT_V2
    src = _v2_store(tmp_path, golden)
    dst = FileSystemStore(str(tmp_path / "v3"), step=0)
    config = convert_checkpoint_v2_to_v3(src, dst)
    assert config["format"] == FORMAT
    model = manager.load_model(dst, device=torch.device("cpu"))
    with torch.no_grad():
        assert torch.equal(model(golden["idx"]), golden["logits"])


@pytest.mark.parametrize("path", GOLDENS, ids=os.path.basename)
def test_load_model_refuses_a_v2_checkpoint_and_points_at_the_converter(manager, tmp_path, path):
    src = _v2_store(tmp_path, _load_golden(path))
    with pytest.raises(ValueError, match="modelcore.convert"):
        manager.load_model(src, device=torch.device("cpu"))


@pytest.mark.parametrize("path", GOLDENS, ids=os.path.basename)
def test_v2_state_under_a_v3_config_also_points_at_the_converter(manager, path):
    """An explicit `config=` skips the stored config, so the state keys are what gives it away."""
    golden = _load_golden(path)

    class _Store:
        def read_model_state(self, map_location=None):
            return golden["state"]

    with pytest.raises(ValueError, match="modelcore.convert"):
        manager.load_model(_Store(), device=torch.device("cpu"), config=ModelConfig.from_dict(golden["config"]))


@pytest.mark.parametrize("path", GOLDENS, ids=os.path.basename)
def test_upgraded_v2_config_builds_and_its_keys_match_the_converted_state(manager, path):
    golden = _load_golden(path)
    config = ModelConfig.from_dict(golden["config"])
    assert config.to_dict()["format"] == FORMAT
    model = build(manager, config)
    assert set(model.state_dict()) == set(convert_state_v2_to_v3(golden["state"]))


def test_convert_cli_round_trip(manager, tmp_path):
    golden = _load_golden(GOLDENS[0])
    src = _v2_store(tmp_path, golden)
    src.update_meta({"val_bpb": 1.25})
    convert_main([str(tmp_path / "v2"), str(tmp_path / "out"), "--step", "0"])
    out = FileSystemStore(str(tmp_path / "out"), step=0)
    assert out.read_meta()["val_bpb"] == 1.25 and out.read_config()["format"] == FORMAT
    with torch.no_grad():
        assert torch.equal(manager.load_model(out, device=torch.device("cpu"))(golden["idx"]), golden["logits"])


def test_converting_a_v3_checkpoint_is_a_no_op_copy(manager, config, tmp_path):
    model = build(manager, config)
    src, dst = FileSystemStore(str(tmp_path / "a"), 0), FileSystemStore(str(tmp_path / "b"), 0)
    manager.save_model(model, src)
    convert_checkpoint_v2_to_v3(src, dst)
    again = manager.load_model(dst, device=torch.device("cpu"))
    for (na, pa), (nb, pb) in zip(model.state_dict().items(), again.state_dict().items()):
        assert na == nb and torch.equal(pa, pb)


# -----------------------------------------------------------------------------
# the config upgrader

def test_v3_to_v2_to_v3_is_the_identity_for_every_flavor(config):
    v3 = config.to_dict()
    assert upgrade_v2_to_v3(downgrade_v3_to_v2(v3)) == v3


def test_upgrade_does_not_mutate_its_input(config):
    v2 = downgrade_v3_to_v2(config.to_dict())
    import copy
    before = copy.deepcopy(v2)
    upgrade_v2_to_v3(v2)
    assert v2 == before


def test_upgrade_to_v3_is_idempotent_and_reaches_v3_from_any_format(config):
    v3 = config.to_dict()
    assert upgrade_to_v3(v3) == v3
    assert upgrade_to_v3(downgrade_v3_to_v2(v3)) == v3


def test_a_gpt_block_becomes_a_block_with_a_lambdas_feature_and_a_value_embed_on_its_mixer():
    d = downgrade_v3_to_v2(FLAVORS["gpt_gated_head"]().to_dict())
    block = d["body"]["blocks"][0]
    assert block["#type"] == "gated_gpt_block" and d["body"]["#type"] == "backout"
    up = upgrade_v2_to_v3(d)
    b = up["body"]["blocks"][-1]  # the last layer carries a value embedding in the gpt flavor
    assert up["body"]["#type"] == "stack" and b["#type"] == "block"
    assert [f["#type"] for f in b["features"]] == ["resid_lambdas"]
    assert b["mixer"]["#type"] == "attention"
    assert [f["#type"] for f in b["mixer"]["features"]] == ["value_embed", "output_gate"]
    assert b["mixer"]["features"][0] == {"#type": "value_embed", "gate_channels": 12}
    assert [f["#type"] for f in up["body"]["features"]] == ["backout"]
    assert "mlp" not in b and b["ffn"]["#type"] == "mlp"


def test_upgrade_rewrites_adapter_targets_and_frozen_fqns():
    v3 = FLAVORS["gpt_lora"]().to_dict()
    v2 = downgrade_v3_to_v2(v3)
    assert v2["adapters"][0]["target"] == "body.blocks.0.attn.c_q"
    assert upgrade_v2_to_v3(v2)["adapters"] == v3["adapters"]


@pytest.mark.parametrize("old,new", [
    ("body.blocks.3.attn.c_q.weight", "body.blocks.3.mixer.c_q.weight"),
    ("body.blocks.3.attn.gate.proj.weight", "body.blocks.3.mixer.features.output_gate.proj.weight"),
    ("body.blocks.3.attn.gate.proj", "body.blocks.3.mixer.features.output_gate.proj"),
    ("body.blocks.3.attn.value_embed.weight", "body.blocks.3.mixer.features.value_embed.embed.weight"),
    ("body.blocks.3.attn.ve_gate.weight", "body.blocks.3.mixer.features.value_embed.gate.weight"),
    ("body.blocks.3.mlp.c_fc.weight", "body.blocks.3.ffn.c_fc.weight"),
    ("body.blocks.3.resid_lambda", "body.blocks.3.features.resid_lambdas.resid_lambda"),
    ("body.blocks.3.x0_lambda", "body.blocks.3.features.resid_lambdas.x0_lambda"),
    ("body.backout_lambda", "body.features.backout.backout_lambda"),
    ("body.blocks.0.attn.c_q.deltas.t0.lora_A.weight", "body.blocks.0.mixer.c_q.deltas.t0.lora_A.weight"),
    ("embedding.wte.weight", "embedding.wte.weight"),
    ("body", "body"),
])
def test_remap_v2_name(old, new):
    assert remap_v2_name(old) == new
    assert remap_v2_name(new) == new  # idempotent: v3 names have no rule


# -----------------------------------------------------------------------------
# a host that keeps emitting v2 component types

def _v2_object_config(v3_config):
    """A ModelConfig whose tree is made of v2 ComponentSpecs -- what a host that has not migrated
    builds directly."""
    v2 = downgrade_v3_to_v2(v3_config.to_dict())
    return ModelConfig(
        sequence_len=v2["sequence_len"], vocab_size=v2["vocab_size"], n_embd=v2["n_embd"],
        pad_vocab_size_to=v2["pad_vocab_size_to"], template=v2["template"],
        shared={k: ComponentSpec.from_dict(s) for k, s in v2["shared"].items()},
        input=ComponentSpec.from_dict(v2["input"]), body=ComponentSpec.from_dict(v2["body"]),
        output=ComponentSpec.from_dict(v2["output"]),
    )


@pytest.mark.parametrize("flavor", ["gpt", "gpt_gated_head", "llama", "llama_kvshare_gated_block"])
def test_a_v2_object_config_builds_the_same_model_as_its_v3_form(manager, flavor):
    v3 = FLAVORS[flavor]()
    v2 = _v2_object_config(v3)
    assert v2.to_dict()["format"] == FORMAT_V2
    assert manager.validate_config(v2).ok
    a, b = build(manager, v3, seed=3), build(manager, v2, seed=3)
    assert b.config.to_dict()["format"] == FORMAT
    assert {k: v.shape for k, v in a.state_dict().items()} == {k: v.shape for k, v in b.state_dict().items()}
    idx = torch.randint(0, v3.vocab_size, (2, 8))
    with torch.no_grad():
        assert torch.equal(a(idx), b(idx))
    assert manager.stats(v2).num_params == manager.stats(v3).num_params
