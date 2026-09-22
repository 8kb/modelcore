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
    whatever means the caller wants, then call .step(token_column) to advance.

    `tokens` is one prompt (list[int]) -- prefilled once and replicated into num_samples identical
    rows -- or several prompts (list[list[int]]), each prefilled separately at batch 1 and copied
    into its own right-ragged row (see KVCache.prefill_row). With P prompts and num_samples=S there
    are P*S rows in prompt-major order: row p*S + s is sample s of prompt p. Every row advances one
    position per step() regardless of how long its own prompt was."""

    @torch.inference_mode()
    def __init__(self, model, manager, tokens, *, num_samples=1, max_tokens=None, device=None):
        self.model = model
        device = device or model.get_device()
        if isinstance(tokens[0], int):
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
            return

        prompts = [list(p) for p in tokens]
        kv_length_hint = (max(map(len, prompts)) + max_tokens) if max_tokens is not None else model.config.sequence_len
        self.kv_cache = manager.new_kv_cache(model, batch_size=len(prompts) * num_samples, seq_len=kv_length_hint, device=device)
        row_logits = []
        for p, prompt in enumerate(prompts):
            # Each prompt is prefilled alone, exactly as above, then freed before the next: only one
            # transient prefill cache is ever alive.
            kv_cache_prefill = manager.new_kv_cache(model, batch_size=1, seq_len=len(prompt), device=device)
            ids = torch.tensor([prompt], dtype=torch.long, device=device)
            row_logits.append(model.forward(ids, kv_cache=kv_cache_prefill)[:, -1, :].expand(num_samples, -1))
            for s in range(num_samples):
                self.kv_cache.prefill_row(p * num_samples + s, kv_cache_prefill)
            del kv_cache_prefill
        self._logits = torch.cat(row_logits, dim=0)  # (P * num_samples, vocab_size)

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

    tokens: prompt token ids (list[int]), or several prompts (list[list[int]]) decoded together.
    With P prompts and num_samples=S each yielded token_column has P*S entries in prompt-major
    order (entry p*S + s is sample s of prompt p) -- see Decoder. terminal_ids: set[int] of ids
    that end a row (e.g. an end-of-turn token, BOS). tools: an ordered list[ToolSpec] -- see
    _advance_row for the start/end/capture priority.

    In the several-prompts form, a row that has hit a terminal id is frozen: still stepped through
    the model (KVCache.advance needs every row to move every step) but fed its own last token and
    skipped by the tool state machine. Without that, a finished row keeps "sampling" garbage and can
    re-open the calculator tool -- one eval per dead row per remaining step of the longest row. The
    single-prompt form never froze rows (its loop exits as soon as all are done) and is unchanged."""
    multi = not isinstance(tokens[0], int)
    prompts = [list(p) for p in tokens] if multi else [tokens]
    assert all(isinstance(p, list) and p and isinstance(p[0], int) for p in prompts), "expecting list of ints (or list of such lists)"
    device = model.get_device()
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)

    decoder = manager.new_decoder(model, tokens, num_samples=num_samples, max_tokens=max_tokens, device=device)
    row_states = [RowState(p.copy()) for p in prompts for _ in range(num_samples)]

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
            if multi and state.completed:
                token_column.append(state.current_tokens[-1])
                token_masks.append(0)
                continue
            is_forced = len(state.forced_tokens) > 0
            token_masks.append(0 if is_forced else 1)
            next_token = state.forced_tokens.popleft() if is_forced else sampled_tokens[i]
            token_column.append(next_token)
            _advance_row(state, next_token, terminal_ids=terminal_ids, tools=tools)

        yield token_column, token_masks
        num_generated += 1
        decoder.step(token_column)


def collect_batch_multi(stream, terminal_ids, prompts, num_samples):
    """collect_batch for several prompts decoded together (see generate_with_tools): returns
    (results, masks), each a list with one group per prompt of num_samples token-id lists, group p
    prefixed with prompts[p]. Row r of the stream is sample r % num_samples of prompt
    r // num_samples (prompt-major)."""
    results = [[p.copy() for _ in range(num_samples)] for p in prompts]
    masks = [[[0] * len(p) for _ in range(num_samples)] for p in prompts]
    completed = [[False] * num_samples for _ in prompts]
    for token_column, token_masks in stream:
        for r, (token, mask) in enumerate(zip(token_column, token_masks)):
            p, s = divmod(r, num_samples)
            if not completed[p][s]:
                if token in terminal_ids:
                    completed[p][s] = True
                else:
                    results[p][s].append(token)
                    masks[p][s].append(mask)
        if all(all(group) for group in completed):
            break
    return results, masks


def collect_batch(stream, terminal_ids, prompt_tokens, num_samples):
    """Drains a generate_with_tools (or any (token_column, token_masks)-yielding) stream into
    (results, masks): each a list of num_samples token-id lists, prompt_tokens-prefixed. A
    terminal token (see terminal_ids) is excluded from the row's own result, and stops that row's
    accumulation, but not the shared decode loop until every row is done."""
    results, masks = collect_batch_multi(stream, terminal_ids, [prompt_tokens], num_samples)
    return results[0], masks[0]
