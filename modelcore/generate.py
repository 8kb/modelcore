"""
Generic (tokenizer-agnostic) autoregressive generation primitives: sampling, a naive
recompute-every-step reference implementation, and Decoder -- a cached prefill+decode primitive
built on ModelManager.new_kv_cache. None of this knows about tokenizers, special tokens, or tool
use; a host application layers those concerns on top (in this repo, nanochat.engine.Engine adds
RowState, the calculator, and chat special tokens), using Decoder for the actual model-stepping.
"""
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
