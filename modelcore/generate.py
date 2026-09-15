"""
Generic (tokenizer-agnostic) autoregressive generation primitives: sampling, a naive
recompute-every-step reference implementation, Decoder -- a cached prefill+decode primitive built
on ModelManager.new_kv_cache -- and generate_with_tools/collect_batch, the tool-use decode loop
both nanochat's and tinylab's Engine used to duplicate line-for-line. None of this knows about
tokenizers or special-token *names*; a host resolves its own special tokens to ids and (for tools)
supplies a ToolSpec per tool -- what the tool's own logic (e.g. nanochat/tinylab's use_calculator,
an eval() sandbox) does with the captured token ids stays entirely on the host side, passed in as
ToolSpec.run. Engine, the class that owns a tokenizer and satisfies benchcore's Generator protocol,
still belongs to the host application; this module only provides the loop it's built on.
"""
from collections import deque
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F


@torch.inference_mode()
def sample_next_token(logits, rng, temperature=1.0, top_k=None):
    """Sample a single next token from given logits of shape (B, vocab_size). Returns (B, 1)."""
    assert temperature >= 0.0, "temperature must be non-negative"
    if temperature == 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        vals, idx = torch.topk(logits, k, dim=-1)
        vals = vals / temperature
        probs = F.softmax(vals, dim=-1)
        choice = torch.multinomial(probs, num_samples=1, generator=rng)
        return idx.gather(1, choice)
    else:
        logits = logits / temperature
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1, generator=rng)


@torch.inference_mode()
def generate_naive(model, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
    """
    Naive autoregressive streaming inference (no KV cache): recomputes the full forward pass at
    every step. Useful as a slow-but-simple reference to check the fast KV-cached Decoder path
    against. To keep this simple, assumes:
    - batch size is 1
    - ids and the yielded tokens are simple Python lists and ints

    Note: sampling here goes through sample_next_token, which (for top_k > 0) draws from the
    renormalized top-k distribution via torch.multinomial. This gives the same distribution as,
    but not necessarily the same draw as, masking to -inf and sampling over the full vocab --
    greedy (temperature=0) is unaffected and remains bit-identical.
    """
    assert isinstance(tokens, list)
    device = model.get_device()
    rng = None
    if temperature > 0:
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)
    ids = torch.tensor([tokens], dtype=torch.long, device=device) # add batch dim
    for _ in range(max_tokens):
        logits = model.forward(ids) # (B, T, vocab_size)
        logits = logits[:, -1, :] # (B, vocab_size)
        next_ids = sample_next_token(logits, rng, temperature, top_k)
        ids = torch.cat((ids, next_ids), dim=1)
        token = next_ids.item()
        yield token


class Decoder:
    """Batch-1 prefill of a prompt, replicated into an num_samples-row KV cache, then stepped one
    position at a time -- the generic half of a cached autoregressive decode loop (the other half,
    e.g. tool-use/forced-token state, belongs to the caller). Construct via
    ModelManager.new_decoder(model, tokens, ...); read .logits, choose a next token per row by
    whatever means the caller wants, then call .step(token_column) to advance."""

    @torch.inference_mode()
    def __init__(self, model, manager, tokens, *, num_samples=1, max_tokens=None, device=None):
        self.model = model
        device = device or model.get_device()
        # 1) Batch-1 prefill of the prompt tokens
        kv_cache_prefill = manager.new_kv_cache(model, batch_size=1, seq_len=len(tokens), device=device)
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        logits = model.forward(ids, kv_cache=kv_cache_prefill)
        self._logits = logits[:, -1, :].expand(num_samples, -1)  # (num_samples, vocab_size)
        # 2) Replicate the KV cache for each sample/row
        kv_length_hint = (len(tokens) + max_tokens) if max_tokens is not None else model.config.sequence_len
        self.kv_cache = manager.new_kv_cache(model, batch_size=num_samples, seq_len=kv_length_hint, device=device)
        self.kv_cache.prefill(kv_cache_prefill)
        del kv_cache_prefill  # no need to keep this memory around

    @property
    def logits(self):
        """Next-token logits, shape (num_samples, vocab_size) -- refreshed by step()."""
        return self._logits

    @torch.inference_mode()
    def step(self, token_column):
        """token_column: num_samples next-token ids, one per row (list[int] or a (num_samples,)
        tensor). Advances every row by one position and returns the new .logits."""
        device = self.model.get_device()
        ids = torch.tensor(token_column, dtype=torch.long, device=device).unsqueeze(1)
        self._logits = self.model.forward(ids, kv_cache=self.kv_cache)[:, -1, :]
        return self._logits


# -----------------------------------------------------------------------------
# Tool-use decode loop. Moved from two identical copies (nanochat.engine.Engine.generate,
# tinylab.engine.Engine.generate) -- everything here is token-id bookkeeping around Decoder; the
# one genuinely host-specific piece (what a captured expression evaluates to) is ToolSpec.run,
# supplied by the caller.


@dataclass
class ToolSpec:
    """One tool a generate_with_tools loop recognizes by its start/end special-token ids. `run`
    receives the token ids captured strictly between start_id and end_id (never including either)
    and returns the tokens to force-inject wrapped in result_start_id/result_end_id, or None to
    inject nothing (e.g. the captured text failed to evaluate -- wrong tool usage isn't fatal)."""
    start_id: int
    end_id: int
    result_start_id: int
    result_end_id: int
    run: Callable[[list], list | None]


class RowState:
    """Per-row state tracking during generation."""
    def __init__(self, current_tokens=None):
        self.current_tokens = current_tokens or []
        self.forced_tokens = deque()
        self.active_tool: ToolSpec | None = None
        self.captured_tokens = []
        self.completed = False


def _advance_row(state: RowState, next_token: int, *, terminal_ids, tools) -> None:
    """Applies one token to `state` in place: terminal detection, and the tool start/end/capture
    state machine (a token matching some tool's start_id, checked first in list order, always
    opens that tool -- even if another tool is already open, matching each of the two ported
    originals' single-tool if/elif/elif priority, generalized to more than one tool). Pure
    token-id bookkeeping, no model/decoder involved -- split out from generate_with_tools so the
    tool-triggering logic is unit-testable without a real forward pass."""
    state.current_tokens.append(next_token)
    if next_token in terminal_ids:
        state.completed = True

    opened = False
    for tool in tools:
        if next_token == tool.start_id:
            state.active_tool = tool
            state.captured_tokens = []
            opened = True
            break
    if not opened and state.active_tool is not None and next_token == state.active_tool.end_id:
        tool = state.active_tool
        state.active_tool = None
        if state.captured_tokens:
            result = tool.run(state.captured_tokens)
            if result is not None:
                state.forced_tokens.append(tool.result_start_id)
                state.forced_tokens.extend(result)
                state.forced_tokens.append(tool.result_end_id)
        state.captured_tokens = []
    elif not opened and state.active_tool is not None:
        state.captured_tokens.append(next_token)


@torch.inference_mode()
def generate_with_tools(model, manager, tokens, *, num_samples=1, max_tokens=None, temperature=1.0,
                         top_k=None, seed=42, terminal_ids, tools=()):
    """Single prefill, then decode num_samples rows from a shared KV cache, recognizing `tools` by
    their start/end ids (see ToolSpec) and stopping a row on any id in `terminal_ids`. Yields
    (token_column, token_masks) per step: mask=0 where a token was tool-forced, 1 if sampled.

    tokens: prompt token ids (list[int]). terminal_ids: set[int] of ids that end a row (e.g. an
    end-of-turn token, BOS). tools: an ordered list[ToolSpec] -- see _advance_row for the
    start/end/capture priority."""
    assert isinstance(tokens, list) and isinstance(tokens[0], int), "expecting list of ints"
    device = model.get_device()
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)

    decoder = manager.new_decoder(model, tokens, num_samples=num_samples, max_tokens=max_tokens, device=device)
    row_states = [RowState(tokens.copy()) for _ in range(num_samples)]

    num_generated = 0
    while True:
        if max_tokens is not None and num_generated >= max_tokens:
            break
        if all(state.completed for state in row_states):
            break

        next_ids = sample_next_token(decoder.logits, rng, temperature, top_k)
        sampled_tokens = next_ids[:, 0].tolist()

        token_column = []
        token_masks = []
        for i, state in enumerate(row_states):
            is_forced = len(state.forced_tokens) > 0
            token_masks.append(0 if is_forced else 1)
            next_token = state.forced_tokens.popleft() if is_forced else sampled_tokens[i]
            token_column.append(next_token)
            _advance_row(state, next_token, terminal_ids=terminal_ids, tools=tools)

        yield token_column, token_masks
        num_generated += 1
        decoder.step(token_column)


def collect_batch(stream, terminal_ids, prompt_tokens, num_samples):
    """Drains a generate_with_tools (or any (token_column, token_masks)-yielding) stream into
    (results, masks): each a list of num_samples token-id lists, prompt_tokens-prefixed. A
    terminal token (see terminal_ids) is excluded from the row's own result, and stops that row's
    accumulation, but not the shared decode loop until every row is done."""
    results = [prompt_tokens.copy() for _ in range(num_samples)]
    masks = [[0] * len(prompt_tokens) for _ in range(num_samples)]
    completed = [False] * num_samples
    for token_column, token_masks in stream:
        for i, (token, mask) in enumerate(zip(token_column, token_masks)):
            if not completed[i]:
                if token in terminal_ids:
                    completed[i] = True
                else:
                    results[i].append(token)
                    masks[i].append(mask)
        if all(completed):
            break
    return results, masks
