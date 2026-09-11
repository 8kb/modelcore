"""
Tests for modelcore/peft/ -- LoRA/DoRA adapters as entries in ModelConfig.adapters/frozen, and
ModelManager's adapter-aware create_optimizer/load_model/stats/merge_adapters/adapters_disabled.

Modelled on test_precision.py's shape (the closest existing precedent: a post-build module-tree
transform whose main risk is silently falling out of modelcore.roles/modelcore.stats).

python -m pytest modelcore/tests/test_peft.py -v
"""
import torch

from modelcore import AdapterSpec, OptimizerHparams
from modelcore.components.linear import Linear
from modelcore.peft import AdapterLinear, find_adapters
from modelcore.roles import collect_param_roles
from modelcore.store import FileSystemStore

from modelcore.tests.conftest import FLAVORS, build


def _lora_config(r=4, alpha=8, freeze_base=True, targets=("body.blocks.0.attn.c_proj",),
                  names=None, dora=False):
    # c_proj (not c_q/c_k/c_v) is the default target deliberately: modelcore zero-initializes
    # every block's final residual projection (CausalSelfAttention.c_proj, MLP.c_proj) as an
    # identity-like init trick, so a change to an *earlier* projection (e.g. c_q) has zero effect
    # on the block's output at init -- it gets multiplied through a zero c_proj.weight before ever
    # reaching the residual sum. A test that wants a trained adapter's effect to be *observable*
    # in the model's output needs a target whose own output isn't masked downstream.
    config = FLAVORS["gpt"]()
    names = names or [f"t{i}" for i in range(len(targets))]
    config.adapters = [
        AdapterSpec(target=t, name=n, type="dora" if dora else "lora", params={"r": r, "alpha": alpha})
        for t, n in zip(targets, names)
    ]
    if freeze_base:
        config.frozen = ["body"]
    return config


# -----------------------------------------------------------------------------
# Structural: AdapterLinear as a modelcore Linear, apply_adapters, roles

def test_adapter_linear_is_a_modelcore_linear():
    assert issubclass(AdapterLinear, Linear)


def test_apply_adapters_converts_target_and_collect_param_roles_succeeds(manager):
    config = _lora_config()
    model = build(manager, config)
    targets = find_adapters(model)
    assert len(targets) == 1
    fqn, module = targets[0]
    assert fqn == "body.blocks.0.attn.c_proj"
    assert isinstance(module, AdapterLinear)
    assert "t0" in module.deltas
    roles = collect_param_roles(model)  # must not raise
    assert "adapter" in roles


def test_create_optimizer_excludes_frozen_base_includes_adapter(manager):
    config = _lora_config(freeze_base=True)
    model = build(manager, config, seed=0)
    optimizer = manager.create_optimizer(model, OptimizerHparams())
    grouped = {id(p) for g in optimizer.param_groups for p in g["params"]}
    trainable = {id(p) for p in model.parameters() if p.requires_grad}
    frozen = {id(p) for p in model.parameters() if not p.requires_grad}
    assert grouped == trainable
    assert not (grouped & frozen)
    assert len(frozen) > 0 and len(trainable) > 0  # sanity: both populations are non-empty here


def test_adapter_params_are_grouped_under_adamw_not_muon(manager):
    config = _lora_config(freeze_base=True)
    model = build(manager, config, seed=0)
    optimizer = manager.create_optimizer(model, OptimizerHparams())
    adapter_module = find_adapters(model)[0][1]
    adapter_param_ids = {id(p) for p in adapter_module.deltas["t0"].parameters()}
    found = False
    for g in optimizer.param_groups:
        if {id(p) for p in g["params"]} & adapter_param_ids:
            assert g["kind"] == "adamw"
            found = True
    assert found, "adapter params should have landed in some optimizer group"


# -----------------------------------------------------------------------------
# Numerics: a fresh adapter is a no-op; a disabled one is skipped, not just near-zero

def test_fresh_lora_adapter_is_an_exact_forward_noop(manager):
    """B is zero-initialized, so a freshly-applied, untrained LoRA adapter must reproduce the
    un-adapted forward *exactly* (not just approximately): B=0 means lora_B's matmul output is
    exactly zero, added to y with no rounding."""
    base_model = build(manager, FLAVORS["gpt"](), seed=0)
    idx = torch.randint(0, base_model.config.vocab_size, (2, 8))
    with torch.no_grad():
        base_out = base_model(idx)

    adapted_model = build(manager, _lora_config(freeze_base=True), seed=0)
    with torch.no_grad():
        adapted_out = adapted_model(idx)
    assert torch.equal(base_out, adapted_out)


def test_fresh_dora_adapter_is_an_approximate_forward_noop(manager):
    """DoRA's magnitude is initialized from the base weight's own row norms, so it's a no-op only
    up to the floating-point round trip of a norm-then-rescale (see modelcore/peft/deltas.py's
    DoRADelta docstring) -- allclose, not equal."""
    base_model = build(manager, FLAVORS["gpt"](), seed=0)
    idx = torch.randint(0, base_model.config.vocab_size, (2, 8))
    with torch.no_grad():
        base_out = base_model(idx)

    adapted_model = build(manager, _lora_config(freeze_base=True, dora=True), seed=0)
    with torch.no_grad():
        adapted_out = adapted_model(idx)
    assert torch.allclose(base_out, adapted_out, atol=1e-3)


def test_disabled_adapter_forward_equals_base_even_when_trained(manager):
    base_model = build(manager, FLAVORS["gpt"](), seed=0)
    idx = torch.randint(0, base_model.config.vocab_size, (2, 8))
    with torch.no_grad():
        base_out = base_model(idx)

    model = build(manager, _lora_config(freeze_base=True), seed=0)  # same seed -> identical base weights
    _, module = find_adapters(model)[0]
    with torch.no_grad():
        module.deltas["t0"].lora_B.weight.normal_(std=0.1)  # simulate a trained (non-zero) adapter

    with torch.no_grad():
        enabled_out = model(idx)
    assert not torch.equal(base_out, enabled_out), "a trained adapter must change the forward output"

    module.set_enabled("t0", False)
    with torch.no_grad():
        disabled_out = model(idx)
    assert torch.equal(base_out, disabled_out)


# -----------------------------------------------------------------------------
# The actual feature: hand-edit a checkpoint's config, reload, no state-dict surgery

def test_hand_edit_disable_then_add_adapter_round_trips_through_reload(manager, tmp_path):
    config = _lora_config(targets=("body.blocks.0.attn.c_proj", "body.blocks.2.mlp.c_proj"), names=["t0", "t1"])
    model = build(manager, config, seed=0)
    # Simulate "trained": perturb both adapters away from their zero-init no-op so enabling/
    # disabling them is numerically observable below.
    with torch.no_grad():
        for _, module in find_adapters(model):
            for delta in module.deltas.values():
                delta.lora_B.weight.normal_(std=0.1)

    store = FileSystemStore(str(tmp_path / "ckpt"), step=0)
    manager.save_model(model, store)
    idx = torch.randint(0, config.vocab_size, (1, 5))

    with torch.no_grad():
        trained_out = manager.load_model(store, device=torch.device("cpu"))(idx)

    # 1. Disable t0 purely via a hand-edited config dict -- no state-dict surgery.
    edited = store.read_config()
    edited["adapters"][0]["enabled"] = False
    disabled_model = manager.load_model(store, device=torch.device("cpu"), config=manager.config_from_dict(edited))
    reloaded_targets = dict(find_adapters(disabled_model))
    assert reloaded_targets["body.blocks.0.attn.c_proj"].enabled["t0"] is False
    assert reloaded_targets["body.blocks.2.mlp.c_proj"].enabled["t1"] is True
    with torch.no_grad():
        disabled_out = disabled_model(idx)
    assert not torch.equal(disabled_out, trained_out)

    # Must match manually disabling t0 on the in-memory (pre-save) model.
    target_module = dict(find_adapters(model))["body.blocks.0.attn.c_proj"]
    target_module.set_enabled("t0", False)
    with torch.no_grad():
        expected = model(idx)
    target_module.set_enabled("t0", True)
    assert torch.equal(disabled_out, expected)

    # 2. Hand-add a brand-new adapter entry (t0/t1 both still enabled, as originally saved); it
    # must initialize to a true no-op, so the reloaded output equals the original trained output.
    edited2 = store.read_config()
    edited2["adapters"].append({
        "target": "body.blocks.0.attn.c_proj", "name": "fresh", "type": "lora",
        "params": {"r": 2, "alpha": 4}, "enabled": True,
    })
    added_model = manager.load_model(store, device=torch.device("cpu"), config=manager.config_from_dict(edited2))
    added_targets = dict(find_adapters(added_model))
    assert "fresh" in added_targets["body.blocks.0.attn.c_proj"].deltas
    with torch.no_grad():
        added_out = added_model(idx)
    assert torch.equal(added_out, trained_out)


# -----------------------------------------------------------------------------
# merge_adapters / adapters_disabled

def test_merge_adapters_output_matches_unmerged_forward(manager):
    config = _lora_config(freeze_base=True)
    model = build(manager, config, seed=0)
    with torch.no_grad():
        find_adapters(model)[0][1].deltas["t0"].lora_B.weight.normal_(std=0.1)
    idx = torch.randint(0, config.vocab_size, (2, 8))
    with torch.no_grad():
        unmerged_out = model(idx)

    merged_count = manager.merge_adapters(model)
    assert merged_count == 1
    assert find_adapters(model) == []
    with torch.no_grad():
        merged_out = model(idx)
    assert torch.allclose(unmerged_out, merged_out, atol=1e-4)


def test_adapters_disabled_context_manager_round_trips_state(manager):
    config = _lora_config(targets=("body.blocks.0.attn.c_q", "body.blocks.2.attn.c_v"), names=["t0", "t1"])
    model = build(manager, config, seed=0)
    _, mod0 = find_adapters(model)[0]
    _, mod1 = find_adapters(model)[1]
    mod1.set_enabled("t1", False)  # one pre-disabled, so restoration isn't trivially "all True"

    idx = torch.randint(0, config.vocab_size, (2, 8))
    with torch.no_grad():
        before = model(idx)
    with manager.adapters_disabled(model):
        assert mod0.enabled["t0"] is False
        assert mod1.enabled["t1"] is False
        with torch.no_grad():
            model(idx)  # must not raise
    assert mod0.enabled["t0"] is True
    assert mod1.enabled["t1"] is False  # restored to its pre-context value, not flipped to True
    with torch.no_grad():
        after = model(idx)
    assert torch.equal(before, after)


def test_adapters_disabled_is_a_noop_with_no_adapters(manager):
    model = build(manager, FLAVORS["llama"]())
    with manager.adapters_disabled(model):
        idx = torch.randint(0, model.config.vocab_size, (2, 8))
        assert torch.isfinite(model(idx)).all()


# -----------------------------------------------------------------------------
# stats: disabled adapters cost no FLOPs but still count as real params

def test_stats_excludes_disabled_adapter_from_matmul_params_not_from_num_params(manager):
    enabled_config = _lora_config(freeze_base=True)
    disabled_config = _lora_config(freeze_base=True)
    disabled_config.adapters[0].enabled = False

    enabled_stats = manager.stats(enabled_config)
    disabled_stats = manager.stats(disabled_config)
    assert disabled_stats.num_matmul_params < enabled_stats.num_matmul_params
    assert disabled_stats.num_params == enabled_stats.num_params


# -----------------------------------------------------------------------------
# fp8 compatibility: enable_fp8 must never clobber an adapter target

def test_fp8_conversion_skips_adapter_targets(manager):
    from modelcore.precision.fp8 import Float8Linear

    config = _lora_config(freeze_base=True)
    model = build(manager, config, seed=0)
    manager.enable_fp8(model, align=1, min_dim=1)  # permissive filter -- would otherwise convert everything
    _, module = find_adapters(model)[0]
    assert isinstance(module, AdapterLinear)
    assert not isinstance(module, Float8Linear)
    assert not any(isinstance(m, Float8Linear) for m in module.deltas.modules())
    # every *other* Linear should still have converted, proving the guard is targeted, not global
    assert any(isinstance(m, Float8Linear) for m in model.modules())


# -----------------------------------------------------------------------------
# validate_config

def test_validate_config_accepts_a_valid_lora_config(manager):
    report = manager.validate_config(_lora_config())
    assert report.ok, report.errors


def test_validate_config_reports_bad_adapter_target(manager):
    config = _lora_config()
    config.adapters[0].target = "body.blocks.0.attn.nonexistent"
    report = manager.validate_config(config)
    assert not report.ok
    assert any("no such module" in e.message for e in report.errors)


def test_validate_config_reports_non_linear_adapter_target(manager):
    config = _lora_config()
    config.adapters[0].target = "body"  # a composer, not a Linear
    report = manager.validate_config(config)
    assert not report.ok
    assert any("not a Linear" in e.message for e in report.errors)


def test_validate_config_reports_unknown_adapter_type(manager):
    config = _lora_config()
    config.adapters[0].type = "nonexistent_type"
    report = manager.validate_config(config)
    assert not report.ok
    assert any("unknown adapter type" in e.message for e in report.errors)


def test_validate_config_reports_bad_adapter_params(manager):
    config = _lora_config()
    config.adapters[0].params = {"r": 4, "alpha": 8, "bogus_kwarg": 1}
    report = manager.validate_config(config)
    assert not report.ok
    assert any("has no such param" in e.message for e in report.errors)


def test_validate_config_reports_duplicate_adapter_name_on_same_target(manager):
    config = _lora_config(targets=("body.blocks.0.attn.c_q", "body.blocks.0.attn.c_q"), names=["dup", "dup"])
    report = manager.validate_config(config)
    assert not report.ok
    assert any("already used on target" in e.message for e in report.errors)


def test_validate_config_reports_bad_frozen_target(manager):
    config = _lora_config()
    config.frozen = ["body.nonexistent"]
    report = manager.validate_config(config)
    assert not report.ok
    assert any("no such module" in e.message for e in report.errors)
