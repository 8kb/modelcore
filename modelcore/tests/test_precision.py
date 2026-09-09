"""
Tests for modelcore/precision/fp8.py and ModelManager.enable_fp8/fp8_disabled.

The core regression this guards: Float8Linear must subclass modelcore.components.linear.Linear,
not a bare nn.Linear, or it silently falls out of the parameter-role protocol
(modelcore.roles.collect_param_roles) and FLOPs/param accounting (modelcore.stats.num_matmul_params)
-- both key off isinstance(module, Linear). Before this fix, converting a model to fp8 then
building its optimizer raised "parameter Float8Linear.weight has no declared role".

The actual FP8 matmul (torch._scaled_mm) needs an FP8-capable CUDA GPU; everything here that only
needs correct module-tree bookkeeping runs on CPU.

python -m pytest modelcore/tests/test_precision.py -v
"""
import torch
import torch.nn as nn

from modelcore.components.linear import Linear
from modelcore.precision.fp8 import Float8Linear, convert_to_float8_training
from modelcore.roles import collect_param_roles
from modelcore.stats import num_matmul_params

from modelcore.tests.conftest import FLAVORS, build


def _convert_all(model):
    """A permissive filter (unlike the hardware-aligned default) so tiny test dims still convert."""
    convert_to_float8_training(model, module_filter_fn=lambda mod, fqn: True)


def test_float8linear_is_a_modelcore_linear():
    assert issubclass(Float8Linear, Linear)
    assert issubclass(Float8Linear, nn.Linear)


def test_convert_then_collect_param_roles_succeeds(manager):
    for flavor in FLAVORS:
        model = build(manager, FLAVORS[flavor]())
        before = num_matmul_params(model)
        _convert_all(model)
        roles = collect_param_roles(model)  # must not raise
        assert sum(len(v) for v in roles.values()) == sum(1 for _ in model.parameters())
        assert num_matmul_params(model) == before, "fp8 conversion must not change matmul-param accounting"


def test_convert_then_create_optimizer_succeeds(manager):
    config = FLAVORS["gpt"]()
    model = build(manager, config)
    _convert_all(model)
    optimizer = manager.create_optimizer(model)  # must not raise
    grouped = sum(len(g["params"]) for g in optimizer.param_groups)
    assert grouped == sum(1 for _ in model.parameters())


def test_enable_fp8_reports_counts(manager):
    config = FLAVORS["gpt"]()
    model = build(manager, config)
    num_linear_before = sum(1 for m in model.modules() if isinstance(m, Linear))
    report = manager.enable_fp8(model, align=1, min_dim=1)  # permissive so tiny dims convert
    assert report.num_linear == num_linear_before
    assert report.num_converted + report.num_skipped == report.num_linear
    assert report.num_converted > 0
    assert sum(1 for m in model.modules() if isinstance(m, Float8Linear)) == report.num_converted


def test_fp8_disabled_round_trips_the_tree(manager):
    config = FLAVORS["gpt"]()
    model = build(manager, config)
    manager.enable_fp8(model, align=1, min_dim=1)
    num_fp8 = sum(1 for m in model.modules() if isinstance(m, Float8Linear))
    assert num_fp8 > 0

    idx = torch.randint(0, config.vocab_size, (2, 8))
    with manager.fp8_disabled(model):
        assert sum(1 for m in model.modules() if isinstance(m, Float8Linear)) == 0
        assert sum(1 for m in model.modules() if isinstance(m, Linear)) >= num_fp8
        logits = model(idx)  # plain Linear forward must still work (fp32/bf16, not fp8 matmul)
        assert torch.isfinite(logits).all()

    # restored afterward
    assert sum(1 for m in model.modules() if isinstance(m, Float8Linear)) == num_fp8


def test_fp8_disabled_is_a_noop_with_no_fp8_modules(manager):
    config = FLAVORS["llama"]()
    model = build(manager, config)
    with manager.fp8_disabled(model):
        idx = torch.randint(0, config.vocab_size, (2, 8))
        logits = model(idx)
        assert torch.isfinite(logits).all()


def test_enable_fp8_default_filter_skips_small_dims(manager):
    """The hardware-aligned default (align=16, min_dim=128) should skip these tiny test dims
    entirely rather than crash -- num_skipped == num_linear."""
    config = FLAVORS["gpt"]()
    model = build(manager, config)
    report = manager.enable_fp8(model)  # default align/min_dim
    assert report.num_converted == 0
    assert report.num_skipped == report.num_linear
