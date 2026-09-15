"""
Tests for modelcore/generate.py: sample_next_token, generate_naive, and Decoder (via
ModelManager.new_decoder) -- the generic, tokenizer-agnostic half of autoregressive generation.
nanochat.engine.Engine layers tool-use/chat-token state on top of the same Decoder; the
Engine-level equivalence check (against real checkpoints) lives in tests/test_goldens.py and
tests/test_generate.py -- this file proves the primitive itself, standalone.

python -m pytest modelcore/tests/test_generate.py -v
"""
import torch

from modelcore.generate import ToolSpec, collect_batch, generate_with_tools, sample_next_token
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
