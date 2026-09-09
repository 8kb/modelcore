"""
Tests for modelcore/generate.py: sample_next_token, generate_naive, and Decoder (via
ModelManager.new_decoder) -- the generic, tokenizer-agnostic half of autoregressive generation.
nanochat.engine.Engine layers tool-use/chat-token state on top of the same Decoder; the
Engine-level equivalence check (against real checkpoints) lives in tests/test_goldens.py and
tests/test_generate.py -- this file proves the primitive itself, standalone.

python -m pytest modelcore/tests/test_generate.py -v
"""
import torch

from modelcore.generate import sample_next_token

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
