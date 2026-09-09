"""
Tests for modelcore/ -- the standalone model subsystem. Fully self-contained: config trees come
from conftest.py's FLAVORS, built directly with ComponentSpec/ModelConfig with no dependency on
any depth-dial or legacy-migration layer (those exist to *produce* such a tree from a depth dial
or an old checkpoint; modelcore only ever consumes the materialized result).

python -m pytest modelcore/tests/test_manager.py -v
"""
import pytest
import torch

from modelcore import ComponentSpec, ModelConfig, OptimizerHparams
from modelcore.components.linear import Linear
from modelcore.roles import collect_param_roles

from modelcore.tests.conftest import FLAVORS, build


# -----------------------------------------------------------------------------
# Generic model behavior, parametrized over every flavor

def test_validate_config_accepts_every_flavor(manager, config):
    report = manager.validate_config(config)
    assert report.ok, report.errors


def test_forward_shape_and_finite(manager, config):
    model = build(manager, config)
    idx = torch.randint(0, config.vocab_size, (2, 8))
    logits = model(idx)
    assert logits.shape == (2, 8, config.vocab_size)
    assert torch.isfinite(logits).all()


def test_loss_is_finite_scalar(manager, config):
    model = build(manager, config)
    idx = torch.randint(0, config.vocab_size, (2, 8))
    targets = torch.randint(0, config.vocab_size, (2, 8))
    loss = model(idx, targets=targets)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_backward_populates_every_gradient(manager, config):
    model = build(manager, config)
    idx = torch.randint(0, config.vocab_size, (2, 8))
    targets = torch.randint(0, config.vocab_size, (2, 8))
    loss = model(idx, targets=targets)
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"{name} got no gradient"
        assert torch.isfinite(p.grad).all(), f"{name} has a non-finite gradient"


def test_params_by_role_partition_matches_total(manager, config):
    stats = manager.stats(config)
    assert sum(stats.params_by_role.values()) == stats.num_params
    model = build(manager, config)
    assert stats.num_params == sum(p.numel() for p in model.parameters())


def test_num_matmul_params_matches_manual_scan(manager, config):
    model = build(manager, config)
    stats = manager.stats(config)
    manual = sum(m.weight.numel() for m in model.modules() if isinstance(m, Linear))
    assert stats.num_matmul_params == manual


def test_llama_flavor_has_no_gpt_residual_topology_extras(manager):
    """Structural proof of the roadmap's "boring baseline" claim: llama's tree has no value
    embeddings, smear, or per-layer resid/x0 scalars -- unlike gpt, which has all four roles."""
    llama_config = FLAVORS["llama"]()
    model = build(manager, llama_config)
    roles = collect_param_roles(model)
    assert "value_embedding" not in roles
    assert "smear" not in roles
    assert "resid_scalar" not in roles
    assert "x0_scalar" not in roles

    gpt_config = FLAVORS["gpt"]()
    gpt_model = build(manager, gpt_config)
    gpt_roles = collect_param_roles(gpt_model)
    assert "value_embedding" in gpt_roles
    assert "smear" in gpt_roles
    assert "resid_scalar" in gpt_roles and "x0_scalar" in gpt_roles
    assert "backout_scalar" in gpt_roles


def test_optimizer_groups_partition_parameters_exactly(manager, config):
    model = build(manager, config)
    optimizer = manager.create_optimizer(model, OptimizerHparams())
    seen = []
    for group in optimizer.param_groups:
        seen.extend(group["params"])
    assert len(seen) == len(set(id(p) for p in seen)), "a parameter appeared in more than one group"
    assert {id(p) for p in seen} == {id(p) for p in model.parameters()}


def test_layer_specs_and_kv_cache_spec_are_consistent(manager, config):
    stats = manager.stats(config)
    assert len(stats.layer_specs) == config.n_layer
    n_kv_heads = {s.n_kv_head for s in stats.layer_specs}
    head_dims = {s.head_dim for s in stats.layer_specs}
    assert len(n_kv_heads) == 1 and len(head_dims) == 1
    slots = {i if s.kv_slot is None else s.kv_slot for i, s in enumerate(stats.layer_specs)}
    assert slots == set(range(len(slots)))
    assert stats.kv_cache_spec["num_kv_slots"] == len(slots)
    assert stats.kv_cache_spec["num_kv_slots"] <= len(stats.layer_specs)


# -----------------------------------------------------------------------------
# ModelManager: create/load/save, seeding, validation

def test_create_model_seed_is_reproducible(manager, config):
    a = build(manager, config, seed=123)
    b = build(manager, config, seed=123)
    for (na, pa), (nb, pb) in zip(a.named_parameters(), b.named_parameters()):
        assert na == nb
        assert torch.equal(pa, pb), f"{na} differs across identically-seeded builds"


def test_save_load_round_trip_preserves_weights_and_forward(manager, config, tmp_path):
    from modelcore.store import FileSystemStore

    model = build(manager, config, seed=0)
    store = FileSystemStore(str(tmp_path / "ckpt"), step=0)
    manager.save_model(model, store)

    reloaded = manager.load_model(store, device=torch.device("cpu"))
    assert reloaded.config == model.config
    original_state = model.state_dict()
    reloaded_state = reloaded.state_dict()
    assert original_state.keys() == reloaded_state.keys()
    for key in original_state:
        assert torch.equal(original_state[key], reloaded_state[key]), f"mismatch in {key}"

    idx = torch.randint(0, config.vocab_size, (1, 5))
    with torch.no_grad():
        assert torch.equal(model(idx), reloaded(idx))


def test_optimizer_save_load_round_trip(manager, config, tmp_path):
    from modelcore.store import FileSystemStore

    model = build(manager, config, seed=0)
    optimizer = manager.create_optimizer(model, OptimizerHparams())
    store = FileSystemStore(str(tmp_path / "ckpt"), step=0)
    manager.save_optimizer(optimizer, store, rank=0)

    reloaded_optimizer = manager.load_optimizer(model, store, rank=0)
    assert reloaded_optimizer is not None
    assert len(reloaded_optimizer.param_groups) == len(optimizer.param_groups)


def test_load_optimizer_returns_none_without_a_saved_shard(manager, config, tmp_path):
    from modelcore.store import FileSystemStore

    model = build(manager, config, seed=0)
    store = FileSystemStore(str(tmp_path / "ckpt"), step=0)
    manager.save_model(model, store)  # no save_optimizer call
    assert manager.load_optimizer(model, store, rank=0) is None


def test_validate_config_reports_every_error_not_just_the_first(manager):
    bad_blocks = [
        ComponentSpec("gpt_block", {
            "layer_idx": 0, "n_head": 3, "n_kv_head": 2, "window": -5, "has_value_embed": False,
            "resid_lambda_init": 1.0, "x0_lambda_init": 0.0,
        }),
    ]
    config = ModelConfig(
        sequence_len=32, vocab_size=128, n_embd=64,
        shared={"rope": ComponentSpec("rotary", {"head_dim": 32})},
        input=ComponentSpec("token_embedding", {"smear": True}),
        body=ComponentSpec("backout", {"backout_layer": 0, "blocks": bad_blocks}),
        output=ComponentSpec("lm_head", {"softcap": 15}),
    )
    report = manager.validate_config(config)
    assert not report.ok
    messages = " ".join(str(e) for e in report.errors)
    assert "divisible by n_head" in messages
    assert "window" in messages
    assert len(report.errors) >= 2, "expected multiple independent errors, not just the first"


def test_validate_config_reports_unknown_component_type(manager):
    config = ModelConfig(
        sequence_len=32, vocab_size=128, n_embd=64,
        shared={}, input=ComponentSpec("nonexistent_embedding", {}),
        body=ComponentSpec("stack", {"blocks": []}),
        output=ComponentSpec("lm_head", {"softcap": 15}),
    )
    report = manager.validate_config(config)
    assert not report.ok
    assert any("unknown component type" in e.message for e in report.errors)


def test_validate_config_reports_missing_input_body_output(manager):
    config = ModelConfig(sequence_len=32, vocab_size=128, n_embd=64)
    report = manager.validate_config(config)
    assert not report.ok
    paths = {e.path for e in report.errors}
    assert {"input", "body", "output"} <= paths


def test_create_model_raises_on_invalid_config(manager):
    config = ModelConfig(sequence_len=32, vocab_size=128, n_embd=64)  # no input/body/output
    with pytest.raises(ValueError):
        manager.create_model(config, device=torch.device("cpu"))


def test_kv_sharing_consumer_without_kv_slot_is_reported(manager):
    blocks = [
        ComponentSpec("plain_block", {"layer_idx": 0, "n_head": 2, "n_kv_head": 2, "window": -1}),
        ComponentSpec("plain_block", {"layer_idx": 1, "n_head": 2, "n_kv_head": 2, "window": -1, "produces_kv": False}),
    ]
    config = ModelConfig(
        sequence_len=32, vocab_size=128, n_embd=64,
        shared={"rope": ComponentSpec("rotary", {"head_dim": 32})},
        input=ComponentSpec("token_embedding", {"smear": False}),
        body=ComponentSpec("stack", {"blocks": blocks}),
        output=ComponentSpec("lm_head", {"softcap": 15}),
    )
    report = manager.validate_config(config)
    assert not report.ok
    assert any("explicit kv_slot" in e.message for e in report.errors)
