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

from modelcore.tests.conftest import FLAVORS, RMS_NORM, build


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
    """Every *trainable* parameter gets a gradient -- not literally every parameter: a frozen
    subtree (config.frozen) or a disabled adapter delta (AdapterLinear.set_enabled) has
    requires_grad=False by construction and is expected to have no .grad, same as it having no
    optimizer group (see modelcore.roles.build_param_groups)."""
    model = build(manager, config)
    idx = torch.randint(0, config.vocab_size, (2, 8))
    targets = torch.randint(0, config.vocab_size, (2, 8))
    loss = model(idx, targets=targets)
    loss.backward()
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
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
    """Every *trainable* parameter is grouped exactly once, and nothing else is: a frozen or
    disabled-adapter parameter must never land in a group, since MuonAdamW.step() dereferences
    p.grad unconditionally and such a parameter's grad is always None (see
    modelcore.roles.build_param_groups)."""
    model = build(manager, config)
    optimizer = manager.create_optimizer(model, OptimizerHparams())
    seen = []
    for group in optimizer.param_groups:
        seen.extend(group["params"])
    assert len(seen) == len(set(id(p) for p in seen)), "a parameter appeared in more than one group"
    trainable = {id(p) for p in model.parameters() if p.requires_grad}
    assert {id(p) for p in seen} == trainable


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


def test_explicit_head_dim_decouples_attention_width_from_n_embd(manager):
    """n_head=8, head_dim=16 -> n_head*head_dim=128, double n_embd=64: c_q/c_k/c_v/c_proj size off
    the wider figure, not off n_embd, and stats/kv_cache_spec agree with the module's own shapes."""
    from modelcore.tests.conftest import _gpt_like
    wide = build(manager, _gpt_like(n_head=8, n_kv_head=8, head_dim=16, n_embd=64))
    narrow = build(manager, _gpt_like(n_head=8, n_kv_head=8, head_dim=8, n_embd=64))  # derived value
    attn = wide.body.blocks[0].attn
    assert attn.head_dim == 16
    assert attn.c_q.weight.shape == (8 * 16, 64)
    assert attn.c_proj.weight.shape == (64, 8 * 16)
    stats_wide = manager.stats(_gpt_like(n_head=8, n_kv_head=8, head_dim=16, n_embd=64))
    stats_narrow = manager.stats(_gpt_like(n_head=8, n_kv_head=8, head_dim=8, n_embd=64))
    assert stats_wide.kv_cache_spec["head_dim"] == 16
    # More heads at a WIDER head_dim than n_embd // n_head genuinely costs more matmul params --
    # the whole point of decoupling: "more heads" is no longer free the way it was when head_dim
    # was always n_embd // n_head.
    assert stats_wide.num_matmul_params > stats_narrow.num_matmul_params


def test_rope_head_dim_mismatch_is_a_runtime_error_not_a_silent_wrong_broadcast(manager):
    """A block's head_dim and shared.rope's own head_dim are never cross-checked by
    validate_config (see modelcore/AGENTS.md) -- documents what actually happens on a mismatch:
    validation passes, and the forward pass raises a shape error rather than silently broadcasting
    something wrong. Not a promise to keep this exact error type/message, just that it's loud."""
    from modelcore.tests.conftest import _gpt_like
    config = _gpt_like(n_head=4, n_kv_head=4, n_embd=64, head_dim=16)  # shared.rope.head_dim == 16
    for block in config.body.params["blocks"]:
        block.params["head_dim"] = 8  # mismatched against shared.rope
    assert manager.validate_config(config).ok
    model = build(manager, config)
    with pytest.raises(RuntimeError):
        model(torch.randint(0, 64, (2, 8)))


def test_head_dim_none_is_bit_identical_to_the_old_derive_only_behavior(manager):
    """head_dim=None (the upgrade path's spelling for a v1 config) must build exactly the model
    n_embd // n_head always built before this param existed -- same shapes, same seeded weights."""
    from modelcore.tests.conftest import _gpt_like
    explicit = _gpt_like(n_head=4, n_kv_head=4, n_embd=64, head_dim=16)  # 64 // 4 == 16
    derived = _gpt_like(n_head=4, n_kv_head=4, n_embd=64, head_dim=16)
    for block in derived.body.params["blocks"]:
        block.params["head_dim"] = None
    a = build(manager, explicit, seed=7)
    b = build(manager, derived, seed=7)
    for (na, pa), (nb, pb) in zip(a.named_parameters(), b.named_parameters()):
        assert na == nb
        assert torch.equal(pa, pb)
    idx = torch.randint(0, 64, (2, 8))
    assert torch.equal(a(idx), b(idx))


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
            "mlp": ComponentSpec("mlp", {"activation": "relu2", "hidden_dim": 256}),
        }),
    ]
    config = ModelConfig(
        sequence_len=32, vocab_size=128, n_embd=64, pad_vocab_size_to=64, template="base",
        shared={"rope": ComponentSpec("rotary", {"head_dim": 32, "over_compute": 10}), "norm": RMS_NORM()},
        input=ComponentSpec("token_embedding", {"smear": True}),
        body=ComponentSpec("backout", {"backout_layer": 0, "backout_lambda_init": 0.2, "blocks": bad_blocks}),
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
        sequence_len=32, vocab_size=128, n_embd=64, pad_vocab_size_to=64, template="base",
        shared={"norm": RMS_NORM()}, input=ComponentSpec("nonexistent_embedding", {}),
        body=ComponentSpec("stack", {"blocks": []}),
        output=ComponentSpec("lm_head", {"softcap": 15}),
    )
    report = manager.validate_config(config)
    assert not report.ok
    assert any("unknown component type" in e.message for e in report.errors)


def test_validate_config_reports_missing_input_body_output(manager):
    config = ModelConfig(sequence_len=32, vocab_size=128, n_embd=64, pad_vocab_size_to=64, template="base")
    report = manager.validate_config(config)
    assert not report.ok
    paths = {e.path for e in report.errors}
    assert {"input", "body", "output"} <= paths


def test_create_model_raises_on_invalid_config(manager):
    config = ModelConfig(sequence_len=32, vocab_size=128, n_embd=64, pad_vocab_size_to=64, template="base")  # no input/body/output
    with pytest.raises(ValueError):
        manager.create_model(config, device=torch.device("cpu"))


def test_kv_sharing_consumer_without_kv_slot_is_reported(manager):
    blocks = [
        ComponentSpec("plain_block", {"layer_idx": 0, "n_head": 2, "n_kv_head": 2, "window": -1, "kv_slot": None, "produces_kv": True, "mlp": ComponentSpec("gated_mlp", {"activation": "silu", "hidden_dim": 256})}),
        ComponentSpec("plain_block", {"layer_idx": 1, "n_head": 2, "n_kv_head": 2, "window": -1, "kv_slot": None, "produces_kv": False, "mlp": ComponentSpec("gated_mlp", {"activation": "silu", "hidden_dim": 256})}),
    ]
    config = ModelConfig(
        sequence_len=32, vocab_size=128, n_embd=64, pad_vocab_size_to=64, template="base",
        shared={"rope": ComponentSpec("rotary", {"head_dim": 32, "over_compute": 10}), "norm": RMS_NORM()},
        input=ComponentSpec("token_embedding", {"smear": False}),
        body=ComponentSpec("stack", {"blocks": blocks}),
        output=ComponentSpec("lm_head", {"softcap": 15}),
    )
    report = manager.validate_config(config)
    assert not report.ok
    assert any("explicit kv_slot" in e.message for e in report.errors)


def test_non_uniform_derived_head_dim_across_blocks_is_reported(manager):
    """head_dim=None derives n_embd // n_head, and n_embd is one global value, so non-uniform
    n_head with head_dim left to derive is exactly non-uniform head_dim -- something
    modelcore.stats.kv_cache_spec() requires uniform (and otherwise only discovers via a raw
    AssertionError at model-build time, not a clean ValidationReport error). Both n_head values
    here (2, 4) individually divide n_embd=64 cleanly, so this isolates the cross-block
    uniformity check from the per-block divisibility check."""
    blocks = [
        ComponentSpec("plain_block", {"layer_idx": 0, "n_head": 2, "n_kv_head": 2, "window": -1, "kv_slot": None, "produces_kv": True, "head_dim": None, "mlp": ComponentSpec("gated_mlp", {"activation": "silu", "hidden_dim": 256})}),
        ComponentSpec("plain_block", {"layer_idx": 1, "n_head": 4, "n_kv_head": 2, "window": -1, "kv_slot": None, "produces_kv": True, "head_dim": None, "mlp": ComponentSpec("gated_mlp", {"activation": "silu", "hidden_dim": 256})}),
    ]
    config = ModelConfig(
        sequence_len=32, vocab_size=128, n_embd=64, pad_vocab_size_to=64, template="base",
        shared={"rope": ComponentSpec("rotary", {"head_dim": 32, "over_compute": 10}), "norm": RMS_NORM()},
        input=ComponentSpec("token_embedding", {"smear": False}),
        body=ComponentSpec("stack", {"blocks": blocks}),
        output=ComponentSpec("lm_head", {"softcap": 15}),
    )
    report = manager.validate_config(config)
    assert not report.ok
    assert any("non-uniform head_dim" in e.message for e in report.errors)


def test_differing_n_head_with_matching_explicit_head_dim_is_not_reported(manager):
    """The whole point of decoupling head_dim from n_embd // n_head: blocks may disagree on
    n_head as long as they agree on head_dim (what the KV cache actually needs uniform).
    n_head 2 vs 4 at a fixed head_dim=16 would derive to 32 vs 16 if left to derive (see the
    sibling non-uniform-derived-head_dim test above) -- stating head_dim explicitly on both
    blocks is what makes this configuration valid."""
    blocks = [
        ComponentSpec("plain_block", {"layer_idx": 0, "n_head": 2, "n_kv_head": 2, "window": -1, "kv_slot": None, "produces_kv": True, "head_dim": 16, "mlp": ComponentSpec("gated_mlp", {"activation": "silu", "hidden_dim": 256})}),
        ComponentSpec("plain_block", {"layer_idx": 1, "n_head": 4, "n_kv_head": 2, "window": -1, "kv_slot": None, "produces_kv": True, "head_dim": 16, "mlp": ComponentSpec("gated_mlp", {"activation": "silu", "hidden_dim": 256})}),
    ]
    config = ModelConfig(
        sequence_len=32, vocab_size=128, n_embd=64, pad_vocab_size_to=64, template="base",
        shared={"rope": ComponentSpec("rotary", {"head_dim": 16, "over_compute": 10}), "norm": RMS_NORM()},
        input=ComponentSpec("token_embedding", {"smear": False}),
        body=ComponentSpec("stack", {"blocks": blocks}),
        output=ComponentSpec("lm_head", {"softcap": 15}),
    )
    report = manager.validate_config(config)
    assert report.ok, report.errors
