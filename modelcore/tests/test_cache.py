"""KVCache: uniform positions, and the right-ragged rows prefill_row creates for multi-prompt decode."""
import pytest
import torch

from modelcore.cache import KVCache
from modelcore.components.rotary import RotaryEmbedding
from modelcore.runtime import DEFAULT_RUNTIME

H, D, T_MAX, SLOTS = 2, 8, 24, 2


def _cache(batch_size, seq_len=T_MAX):
    return KVCache(batch_size=batch_size, num_heads=H, seq_len=seq_len, head_dim=D,
                   num_kv_slots=SLOTS, device="cpu", dtype=torch.float32)


def _prefilled(length, seed, with_state=False):
    """A batch=1 cache holding `length` random positions, like a finished prompt prefill."""
    g = torch.Generator().manual_seed(seed)
    c = _cache(1, seq_len=length)
    c.k_cache.copy_(torch.randn(c.k_cache.shape, generator=g))
    c.v_cache.copy_(torch.randn(c.v_cache.shape, generator=g))
    c.advance(length)
    if with_state:
        c.state["prev_embedding"] = torch.randn(1, 1, 6, generator=g)
    return c


def test_uniform_cache_reports_its_single_position():
    c = _cache(3)
    assert c.get_pos() == c.uniform_pos() == 0
    c.advance(5)
    assert c.get_pos() == c.uniform_pos() == 5
    c.reset()
    assert c.get_pos() == 0


def test_prefill_replicates_one_prompt_into_every_row_and_stays_uniform():
    src = _prefilled(7, seed=1)
    dst = _cache(4)
    dst.prefill(src)
    assert dst.get_pos() == 7
    for row in range(4):
        assert torch.equal(dst.k_cache[:, row, :7], src.k_cache[:, 0, :7])


def test_prefill_row_copies_only_its_row_and_sets_only_its_length():
    a, b = _prefilled(5, seed=1), _prefilled(11, seed=2)
    dst = _cache(3)
    dst.prefill_row(1, a)
    assert dst.cache_seqlens.tolist() == [0, 5, 0]
    assert torch.equal(dst.k_cache[:, 1, :5], a.k_cache[:, 0, :5])
    assert dst.k_cache[:, 0].abs().sum() == 0 and dst.k_cache[:, 2].abs().sum() == 0
    dst.prefill_row(2, b)
    assert dst.cache_seqlens.tolist() == [0, 5, 11]
    assert torch.equal(dst.v_cache[:, 2, :11], b.v_cache[:, 0, :11])


def test_get_pos_raises_on_a_ragged_cache_instead_of_answering_for_row_zero():
    dst = _cache(2)
    dst.prefill_row(0, _prefilled(5, seed=1))
    dst.prefill_row(1, _prefilled(9, seed=2))
    assert dst.uniform_pos() is None
    with pytest.raises(AssertionError, match="ragged"):
        dst.get_pos()


def test_equal_length_rows_are_uniform_again():
    dst = _cache(2)
    dst.prefill_row(0, _prefilled(6, seed=1))
    dst.prefill_row(1, _prefilled(6, seed=2))
    assert dst.uniform_pos() == dst.get_pos() == 6


def test_advance_preserves_the_differences_between_rows():
    dst = _cache(3)
    for row, n in enumerate([3, 8, 5]):
        dst.prefill_row(row, _prefilled(n, seed=row))
    for _ in range(4):
        dst.advance(1)
    assert dst.cache_seqlens.tolist() == [7, 12, 9]
    assert dst.uniform_pos() is None


def test_prefill_row_carries_per_row_state():
    dst = _cache(2)
    a, b = _prefilled(4, seed=1, with_state=True), _prefilled(6, seed=2, with_state=True)
    dst.prefill_row(0, a)
    dst.prefill_row(1, b)
    got = dst.state["prev_embedding"]
    assert got.shape == (2, 1, 6)
    assert torch.equal(got[0], a.state["prev_embedding"][0])
    assert torch.equal(got[1], b.state["prev_embedding"][0])


def test_prefill_row_refuses_a_filled_row_a_wide_source_and_a_too_long_prompt():
    dst = _cache(2)
    dst.prefill_row(0, _prefilled(4, seed=1))
    with pytest.raises(AssertionError, match="non-empty"):
        dst.prefill_row(0, _prefilled(4, seed=2))
    with pytest.raises(AssertionError, match="batch=1"):
        dst.prefill_row(1, _cache(2))
    with pytest.raises(AssertionError, match="exceeds cache length"):
        dst.prefill_row(1, _prefilled(T_MAX + 1, seed=3))


def test_rotary_gathers_each_rows_own_window_when_ragged():
    rot = RotaryEmbedding(head_dim=D, sequence_len=32, over_compute=2)
    rot.init_weights()
    dst = _cache(3)
    lens = [3, 12, 7]
    for row, n in enumerate(lens):
        dst.prefill_row(row, _prefilled(n, seed=row))
    cos, sin = rot._cos_sin(1, dst)
    assert cos.shape == (3, 1, 1, D // 2)
    for row, n in enumerate(lens):
        assert torch.equal(cos[row], rot.cos[:, n:n + 1][0])
        assert torch.equal(sin[row], rot.sin[:, n:n + 1][0])
    # multi-token windows too (chunked decode)
    cos3, _ = rot._cos_sin(3, dst)
    for row, n in enumerate(lens):
        assert torch.equal(cos3[row], rot.cos[0, n:n + 3])


def test_rotary_uniform_cache_keeps_the_broadcast_window():
    rot = RotaryEmbedding(head_dim=D, sequence_len=32, over_compute=2)
    rot.init_weights()
    c = _cache(3)
    c.advance(9)
    cos, _ = rot._cos_sin(2, c)
    assert cos.shape == (1, 2, 1, D // 2)
    assert torch.equal(cos, rot.cos[:, 9:11])
