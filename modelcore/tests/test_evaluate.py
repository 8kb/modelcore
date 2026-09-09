"""
Tests for modelcore.evaluate.evaluate_bpb / ModelManager.evaluate_bpb. Most tests here drive the
function against a FakeModel with a hand-picked per-token loss table, so the expected
bits-per-byte can be computed by hand -- no real model, no dependency on a datacore loader (a
plain iterable of (x, y[, state]) tuples is the whole contract). One integration test at the
bottom exercises the real path against a conftest-built model, to prove the wrapper actually
reaches a real Model's forward().

python -m pytest modelcore/tests/test_evaluate.py -v
"""
import math

import numpy as np
import pytest
import torch

from modelcore.tests.conftest import build


class FakeModel:
    """Stands in for modelcore.Model: get_device() and a forward that returns a preset
    (B, T) loss tensor, ignoring x/idx entirely (fine since these tests never pass
    bos_token_id, so build_doc_args is never invoked and x's content is never read)."""

    def __init__(self, loss2d, device="cpu"):
        self.loss2d = loss2d
        self._device = torch.device(device)

    def get_device(self):
        return self._device

    def __call__(self, x, y, loss_reduction='none', doc_args=None):
        assert loss_reduction == 'none'
        return self.loss2d


def _one_batch(x, y):
    """A single-batch iterable, matching the (x, y, *_) unpacking evaluate_bpb does -- an
    infinite repeat is fine since steps=1 in every test below."""
    while True:
        yield x, y, {"ignored": "resume state"}


def test_matches_hand_computed_bpb(manager):
    # 2 rows x 3 cols, all real (non-ignored) targets, all with a positive byte length.
    loss2d = torch.tensor([[1.0, 2.0, 0.5], [0.25, 1.5, 3.0]])
    y = torch.tensor([[1, 2, 3], [4, 5, 6]])
    token_bytes = [0, 1, 2, 1, 3, 1, 2]  # index 0 unused (no token id 0 in y here)
    model = FakeModel(loss2d)
    bpb = manager.evaluate_bpb(model, _one_batch(None, y), steps=1, token_bytes=token_bytes)

    expected_nats = sum(loss2d.flatten().tolist())  # every target's byte length is > 0
    expected_bytes = sum(token_bytes[t] for t in y.flatten().tolist())
    assert bpb == pytest.approx(expected_nats / (math.log(2) * expected_bytes))


def test_zero_byte_tokens_excluded_from_both_sums(manager):
    loss2d = torch.tensor([[1.0, 2.0, 5.0]])
    y = torch.tensor([[1, 2, 3]])  # token 3 has byte length 0 -- a "special token"
    token_bytes = [0, 1, 2, 0]
    model = FakeModel(loss2d)
    bpb = manager.evaluate_bpb(model, _one_batch(None, y), steps=1, token_bytes=token_bytes)

    # position 2's loss (5.0) must NOT be counted, since its target's byte length is 0.
    expected_nats = 1.0 + 2.0
    expected_bytes = 1 + 2
    assert bpb == pytest.approx(expected_nats / (math.log(2) * expected_bytes))


def test_ignore_index_targets_excluded(manager):
    loss2d = torch.tensor([[1.0, 2.0, 9.0]])
    y = torch.tensor([[1, 2, -1]])  # -1 is the ignore_index sentinel
    token_bytes = [0, 1, 2]
    model = FakeModel(loss2d)
    bpb = manager.evaluate_bpb(model, _one_batch(None, y), steps=1, token_bytes=token_bytes)

    expected_nats = 1.0 + 2.0
    expected_bytes = 1 + 2
    assert bpb == pytest.approx(expected_nats / (math.log(2) * expected_bytes))


def test_all_zero_byte_table_returns_inf(manager):
    loss2d = torch.tensor([[1.0, 2.0]])
    y = torch.tensor([[1, 2]])
    token_bytes = [0, 0, 0]
    model = FakeModel(loss2d)
    bpb = manager.evaluate_bpb(model, _one_batch(None, y), steps=1, token_bytes=token_bytes)
    assert bpb == float('inf')


@pytest.mark.parametrize("as_", [list, np.array, torch.tensor])
def test_accepts_list_numpy_or_tensor_token_bytes(manager, as_):
    loss2d = torch.tensor([[1.0, 2.0, 0.5]])
    y = torch.tensor([[1, 2, 3]])
    values = [0, 1, 2, 3]
    model = FakeModel(loss2d)
    bpb = manager.evaluate_bpb(model, _one_batch(None, y), steps=1, token_bytes=as_(values))

    expected_nats = sum(loss2d.flatten().tolist())
    expected_bytes = sum(values[t] for t in y.flatten().tolist())
    assert bpb == pytest.approx(expected_nats / (math.log(2) * expected_bytes))


def test_multiple_steps_accumulate(manager):
    """steps=2 over two distinct batches sums both, rather than only reading the first."""
    y1 = torch.tensor([[1, 2]])
    y2 = torch.tensor([[3, 4]])
    loss1 = torch.tensor([[1.0, 1.0]])
    loss2 = torch.tensor([[2.0, 2.0]])
    token_bytes = [0, 1, 1, 1, 1]

    class TwoStepModel:
        def __init__(self):
            self._device = torch.device("cpu")
            self._n = 0

        def get_device(self):
            return self._device

        def __call__(self, x, y, loss_reduction='none', doc_args=None):
            self._n += 1
            return loss1 if self._n == 1 else loss2

        def batches(self):
            yield None, y1, {}
            yield None, y2, {}
            while True:
                yield None, y2, {}  # evaluate_bpb only ever calls next() steps times

    model = TwoStepModel()
    bpb = manager.evaluate_bpb(model, model.batches(), steps=2, token_bytes=token_bytes)
    expected_nats = 1.0 + 1.0 + 2.0 + 2.0
    expected_bytes = 4  # four target tokens total, each byte length 1
    assert bpb == pytest.approx(expected_nats / (math.log(2) * expected_bytes))


def test_evaluate_bpb_runs_against_a_real_model(manager, config):
    """Integration smoke: the ModelManager.evaluate_bpb wrapper reaches a real Model.forward()
    and produces a finite, non-negative number, for every conftest flavor."""
    model = build(manager, config)
    vocab_size = config.vocab_size
    x = torch.randint(0, vocab_size, (2, 8))
    y = torch.randint(0, vocab_size, (2, 8))
    token_bytes = [1] * vocab_size  # uniform, non-zero -- every token counts
    bpb = manager.evaluate_bpb(model, _one_batch(x, y), steps=1, token_bytes=token_bytes)
    assert bpb == pytest.approx(bpb) and bpb >= 0 and math.isfinite(bpb)
