"""
Tests for modelcore/generate.py: sample_next_token, generate_naive, and Decoder (via
ModelManager.new_decoder) -- the generic, tokenizer-agnostic half of autoregressive generation.
nanochat.engine.Engine layers tool-use/chat-token state on top of the same Decoder; the
Engine-level equivalence check (against real checkpoints) lives in tests/test_goldens.py and
tests/test_generate.py -- this file proves the primitive itself, standalone.

python -m pytest modelcore/tests/test_generate.py -v
"""
import torch

import pytest

import modelcore.generate as generate_module
import modelcore.kernels.flash_attn as fa_module
from modelcore.generate import ToolSpec, collect_batch, collect_batch_multi, generate_with_tools, sample_next_token
from modelcore.generate import RowState, _advance_row

from modelcore.tests.conftest import FLAVORS, build


def test_sample_next_token_greedy_is_argmax():
    logits = torch.tensor([[1.0, 5.0, 2.0], [3.0, 0.5, 9.0]])
    next_ids = sample_next_token(logits, rng=None, temperature=0.0)
    assert next_ids.tolist() == [[1], [2]]


def test_sample_next_token_is_deterministic_at_a_fixed_seed():
    logits = torch.randn(4, 16)
    rng1 = torch.Generator().manual_seed(0)
    rng2 = torch.Generator().manual_seed(0)
    a = sample_next_token(logits, rng1, temperature=1.0, top_k=4)
    b = sample_next_token(logits, rng2, temperature=1.0, top_k=4)
    assert torch.equal(a, b)


def test_decoder_matches_generate_naive_at_temperature_zero(manager):
    for flavor in FLAVORS:
        config = FLAVORS[flavor]()
        model = build(manager, config)
        model.eval()
        prompt = [1, 2, 3, 4]
        max_tokens = 6

        from modelcore.generate import generate_naive
        naive_tokens = list(generate_naive(model, prompt, max_tokens=max_tokens, temperature=0.0))

        decoder = manager.new_decoder(model, prompt, num_samples=1, max_tokens=max_tokens)
        cached_tokens = []
        for _ in range(max_tokens):
            next_id = sample_next_token(decoder.logits, rng=None, temperature=0.0)
            token = next_id.item()
            cached_tokens.append(token)
            decoder.step([token])

        assert cached_tokens == naive_tokens, f"{flavor}: naive vs Decoder disagree at temperature=0"


def test_decoder_logits_shape_and_step_updates_them(manager):
    config = FLAVORS["gpt"]()
    model = build(manager, config)
    model.eval()
    prompt = [1, 2, 3]
    decoder = manager.new_decoder(model, prompt, num_samples=3, max_tokens=4)
    assert decoder.logits.shape == (3, config.vocab_size)
    before = decoder.logits.clone()
    new_logits = decoder.step([1, 2, 3])
    assert new_logits is decoder.logits
    assert new_logits.shape == (3, config.vocab_size)
    assert not torch.equal(before, decoder.logits), "step() should advance the cache and change logits"


# -----------------------------------------------------------------------------
# _advance_row: the tool start/end/capture state machine, pure token-id bookkeeping with no model
# involved -- see generate_with_tools' docstring for why this is split out and directly testable.

TERMINAL = {99}
START, END, RESULT_START, RESULT_END = 10, 11, 20, 21


def _echo_tool(captured):
    """A ToolSpec.run that just echoes the captured tokens back -- enough to prove routing without
    needing a real tokenizer/calculator."""
    return list(captured)


def test_advance_row_marks_completed_on_terminal_id():
    state = RowState([1, 2, 3])
    _advance_row(state, 99, terminal_ids=TERMINAL, tools=())
    assert state.completed
    assert state.current_tokens == [1, 2, 3, 99]


def test_advance_row_runs_tool_and_forces_wrapped_result():
    tool = ToolSpec(START, END, RESULT_START, RESULT_END, run=_echo_tool)
    state = RowState([])
    for token in (START, 5, 6, END):
        _advance_row(state, token, terminal_ids=TERMINAL, tools=[tool])
    assert state.active_tool is None
    assert list(state.forced_tokens) == [RESULT_START, 5, 6, RESULT_END]


def test_advance_row_tool_returning_none_forces_nothing():
    tool = ToolSpec(START, END, RESULT_START, RESULT_END, run=lambda captured: None)
    state = RowState([])
    for token in (START, 7, END):
        _advance_row(state, token, terminal_ids=TERMINAL, tools=[tool])
    assert list(state.forced_tokens) == []


def test_advance_row_empty_capture_never_calls_run():
    calls = []
    tool = ToolSpec(START, END, RESULT_START, RESULT_END, run=lambda captured: calls.append(captured) or [1])
    state = RowState([])
    for token in (START, END):  # end immediately follows start -- nothing captured
        _advance_row(state, token, terminal_ids=TERMINAL, tools=[tool])
    assert calls == []
    assert list(state.forced_tokens) == []


def test_advance_row_nested_start_resets_capture():
    """A second start_id while already inside the tool re-opens it and drops what was captured so
    far -- matches the two ported originals' if/elif/elif priority (start always wins)."""
    tool = ToolSpec(START, END, RESULT_START, RESULT_END, run=_echo_tool)
    state = RowState([])
    for token in (START, 1, 2, START, 3, END):
        _advance_row(state, token, terminal_ids=TERMINAL, tools=[tool])
    assert list(state.forced_tokens) == [RESULT_START, 3, RESULT_END]


def test_advance_row_two_tools_start_id_selects_the_right_one():
    tool_a = ToolSpec(10, 11, 20, 21, run=lambda c: ["a"] + c)
    tool_b = ToolSpec(30, 31, 40, 41, run=lambda c: ["b"] + c)
    state = RowState([])
    for token in (30, 5, 31):
        _advance_row(state, token, terminal_ids=TERMINAL, tools=[tool_a, tool_b])
    assert list(state.forced_tokens) == [40, "b", 5, 41]


# -----------------------------------------------------------------------------
# collect_batch

def test_collect_batch_drops_terminal_and_stops_when_all_rows_complete():
    stream = [
        ([1, 5], [1, 1]),
        ([99, 6], [1, 1]),   # row 0 hits terminal and stops accumulating; row 1 keeps going
        ([0, 99], [0, 1]),   # row 0's forced 0 must be ignored (already completed); row 1 ends
        ([0, 0], [1, 1]),    # never reached -- both rows completed on the previous step
    ]
    results, masks = collect_batch(iter(stream), TERMINAL, prompt_tokens=[7], num_samples=2)
    assert results == [[7, 1], [7, 5, 6]]
    assert masks == [[0, 1], [0, 1, 1]]


def test_generate_with_tools_respects_max_tokens_with_no_tools(manager):
    config = FLAVORS["gpt"]()
    model = build(manager, config)
    model.eval()
    prompt = [1, 2, 3]
    max_tokens = 5
    stream = list(generate_with_tools(model, manager, prompt, num_samples=2, max_tokens=max_tokens,
                                       temperature=0.0, terminal_ids=set(), tools=()))
    assert len(stream) == max_tokens
    for token_column, token_masks in stream:
        assert len(token_column) == 2
        assert token_masks == [1, 1]  # nothing forced -- every step is a real sample


# -----------------------------------------------------------------------------
# Multi-prompt (ragged) batched decode: several DIFFERENT prompts in one batch must produce, at
# temperature 0, exactly what each prompt produces decoded alone. Right-ragged rows are what make
# that hold: a row's KV starts at index 0, so its RoPE positions and sliding window are its own.

# Lengths straddle the flavors' window (8): a prompt shorter than it, one near it, ones past it,
# and max_tokens large enough that every window actually slides during decode.
def _build_awake(manager, config, seed=0):
    """build(), but with every parameter init left at exactly zero given small random values.

    The init zeroes each attention c_proj, each MLP down-projection and the smear lambda_ (a
    zero-init residual design), so a freshly built model's logits are a function of the token
    embedding alone: attention, RoPE, the sliding window and the smear state cannot affect them.
    Any test of those has to wake them first, or it passes whatever the code does -- mutation
    testing showed exactly that (a wrong RoPE position and a dropped smear state both survived)."""
    model = build(manager, config, seed=seed)
    g = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        for p in model.parameters():
            if p.numel() and p.abs().sum() == 0:
                p.copy_(torch.randn(p.shape, generator=g) * 0.1)
    return model


RAGGED_PROMPTS = [[1, 2], [3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15, 16], [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]]
NO_TERMINAL = frozenset({10**6})  # outside the vocab: no row ever terminates on it


def _alone(manager, model, prompt, num_samples=1, max_tokens=12, terminal=NO_TERMINAL):
    stream = generate_with_tools(model, manager, prompt, num_samples=num_samples, max_tokens=max_tokens,
                                  temperature=0.0, terminal_ids=terminal)
    return collect_batch(stream, terminal, prompt, num_samples)


def _together(manager, model, prompts, num_samples=1, max_tokens=12, terminal=NO_TERMINAL):
    stream = generate_with_tools(model, manager, prompts, num_samples=num_samples, max_tokens=max_tokens,
                                  temperature=0.0, terminal_ids=terminal)
    return collect_batch_multi(stream, terminal, prompts, num_samples)


@pytest.mark.parametrize("force_ragged", [True, False], ids=["same-arithmetic", "as-in-production"])
@pytest.mark.parametrize("flavor", list(FLAVORS))
def test_multi_prompt_decode_matches_each_prompt_alone_at_temperature_zero(manager, monkeypatch, flavor, force_ragged):
    """force_ragged=True runs BOTH sides through the masked branch, so any difference is a real bug.
    False is what production does (a lone prompt takes the uniform sliced path), i.e. the claim that
    actually matters to a user comparing batched eval against the unbatched numbers they have."""
    monkeypatch.setattr(fa_module, "_force_ragged", force_ragged)
    model = _build_awake(manager, FLAVORS[flavor]())
    model.eval()
    results, masks = _together(manager, model, RAGGED_PROMPTS)
    for p, prompt in enumerate(RAGGED_PROMPTS):
        alone, alone_masks = _alone(manager, model, prompt)
        assert results[p] == alone, f"{flavor}: prompt {p} (len {len(prompt)}) differs batched vs alone"
        assert masks[p] == alone_masks
        assert len(results[p][0]) == len(prompt) + 12


def test_one_prompt_through_the_multi_form_equals_the_single_form(manager):
    model = _build_awake(manager, FLAVORS["gpt"]())
    model.eval()
    prompt = RAGGED_PROMPTS[1]
    single, single_masks = _alone(manager, model, prompt, num_samples=2)
    multi, multi_masks = _together(manager, model, [prompt], num_samples=2)
    assert multi == [single] and multi_masks == [single_masks]


def test_multi_prompt_rows_are_prompt_major(manager):
    model = _build_awake(manager, FLAVORS["llama"]())
    model.eval()
    prompts = [RAGGED_PROMPTS[0], RAGGED_PROMPTS[2]]
    results, _ = _together(manager, model, prompts, num_samples=3, max_tokens=5)
    assert len(results) == 2 and all(len(group) == 3 for group in results)
    for p, prompt in enumerate(prompts):
        for row in results[p]:
            assert row[:len(prompt)] == prompt
        # temperature 0: every sample of one prompt is the same continuation
        assert results[p][0] == results[p][1] == results[p][2]


def test_multi_prompt_cache_is_ragged_and_each_row_advances_one_per_step(manager):
    model = _build_awake(manager, FLAVORS["llama"]())
    model.eval()
    decoder = manager.new_decoder(model, RAGGED_PROMPTS, num_samples=1, max_tokens=6)
    lens = [len(p) for p in RAGGED_PROMPTS]
    assert decoder.kv_cache.cache_seqlens.tolist() == lens
    assert decoder.kv_cache.uniform_pos() is None
    assert decoder.logits.shape == (len(RAGGED_PROMPTS), model.config.vocab_size)
    decoder.step([1] * len(RAGGED_PROMPTS))
    decoder.step([1] * len(RAGGED_PROMPTS))
    assert decoder.kv_cache.cache_seqlens.tolist() == [n + 2 for n in lens]


def test_a_finished_row_is_frozen_and_never_disturbs_the_others(manager, monkeypatch):
    """Pick a terminal id that prompt A greedily emits on its 3rd token. In a batch, A must stop
    after 2 tokens, B must decode all 12 exactly as it would alone, and the tool state machine
    must run for A only until it finished -- not for every remaining step of B."""
    model = _build_awake(manager, FLAVORS["llama_kvshare_win"]())
    model.eval()
    a, b = RAGGED_PROMPTS[0], RAGGED_PROMPTS[3]
    terminal = frozenset({_alone(manager, model, a, max_tokens=8)[0][0][len(a) + 2]})
    b_alone, _ = _alone(manager, model, b, terminal=terminal)
    assert terminal.isdisjoint(b_alone[0][len(b):]), "pick prompts where B never hits A's terminal"

    a_two_tokens = _alone(manager, model, a, max_tokens=2)[0][0]   # reference, before counting starts

    calls = []
    real = generate_module._advance_row
    monkeypatch.setattr(generate_module, "_advance_row",
                        lambda state, token, **kw: (calls.append(id(state)), real(state, token, **kw))[1])
    results, _ = _together(manager, model, [a, b], terminal=terminal)

    assert results[0][0] == a_two_tokens
    assert len(results[0][0]) == len(a) + 2                    # stopped at its terminal
    assert results[1] == b_alone                                # B unaffected by A finishing
    per_row = sorted(calls.count(i) for i in set(calls))
    assert per_row == [3, 12], f"expected the finished row advanced 3 times and the long row 12, got {per_row}"


@pytest.mark.parametrize("force_ragged", [True, False], ids=["same-arithmetic", "as-in-production"])
@pytest.mark.parametrize("flavor", list(FLAVORS))
def test_multi_prompt_logits_match_each_prompt_alone_at_every_step(manager, monkeypatch, flavor, force_ragged):
    """Token equality alone is a weak check here: a randomly initialized tiny model's greedy tokens
    barely depend on attention details, so a wrong RoPE position or a missing smear state can leave
    every argmax unchanged. Logits can't hide it. Fixed (non-argmax) tokens are fed to every decoder
    so the batched and alone paths can never diverge onto different sequences."""
    monkeypatch.setattr(fa_module, "_force_ragged", force_ragged)
    model = _build_awake(manager, FLAVORS[flavor]())
    model.eval()
    steps = 12
    together = manager.new_decoder(model, RAGGED_PROMPTS, max_tokens=steps)
    alone = [manager.new_decoder(model, p, max_tokens=steps) for p in RAGGED_PROMPTS]
    for step in range(steps + 1):
        for i, decoder in enumerate(alone):
            torch.testing.assert_close(together.logits[i], decoder.logits[0], atol=1e-4, rtol=1e-4,
                                        msg=lambda m: f"{flavor} step {step} prompt {i}: {m}")
        if step == steps:
            break                       # the cache holds exactly `steps` decoded positions
        column = [(step * 7 + i * 3) % 100 + 1 for i in range(len(alone))]
        together.step(column)
        for token, decoder in zip(column, alone):
            decoder.step([token])
