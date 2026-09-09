"""
Shared config-tree builders for modelcore's test suite. Builds trees directly with
ComponentSpec/ModelConfig -- no dependency on nanochat.architectures (that package exists to
*produce* such a tree from a depth dial or an old checkpoint; modelcore's own tests only ever
consume an already-materialized one). These four flavors mirror the four presets
nanochat.architectures.presets.expand() knows how to derive, so a bug that trips one of these
generic tests would trip the corresponding preset too.
"""
import pytest
import torch

from modelcore import ComponentSpec, ModelConfig, ModelManager


def _gpt_like(n_layer=4, n_head=2, n_kv_head=2, n_embd=64, head_dim=32, vocab_size=128, sequence_len=32, window=-1):
    blocks = [
        ComponentSpec("gpt_block", {
            "layer_idx": i, "n_head": n_head, "n_kv_head": n_kv_head, "window": window,
            "has_value_embed": (i % 2 == (n_layer - 1) % 2),
            "resid_lambda_init": 1.15 - 0.10 * i / max(n_layer - 1, 1),
            "x0_lambda_init": 0.20 - 0.15 * i / max(n_layer - 1, 1),
        })
        for i in range(n_layer)
    ]
    return ModelConfig(
        sequence_len=sequence_len, vocab_size=vocab_size, n_embd=n_embd,
        shared={"rope": ComponentSpec("rotary", {"head_dim": head_dim})},
        input=ComponentSpec("token_embedding", {"smear": True}),
        body=ComponentSpec("backout", {"backout_layer": n_layer // 2, "backout_lambda_init": 0.2, "blocks": blocks}),
        output=ComponentSpec("lm_head", {"softcap": 15}),
    )


def _plain_like(n_layer=4, n_head=2, n_kv_head=2, n_embd=64, head_dim=32, vocab_size=128, sequence_len=32,
                 window=-1, kv_slots=None):
    blocks = []
    for i in range(n_layer):
        params = {"layer_idx": i, "n_head": n_head, "n_kv_head": n_kv_head, "window": window}
        if kv_slots is not None:
            params["kv_slot"] = kv_slots[i]
            params["produces_kv"] = kv_slots[i] == i
        blocks.append(ComponentSpec("plain_block", params))
    return ModelConfig(
        sequence_len=sequence_len, vocab_size=vocab_size, n_embd=n_embd,
        shared={"rope": ComponentSpec("rotary", {"head_dim": head_dim})},
        input=ComponentSpec("token_embedding", {"smear": False}),
        body=ComponentSpec("stack", {"blocks": blocks}),
        output=ComponentSpec("lm_head", {"softcap": 15}),
    )


FLAVORS = {
    "gpt": lambda: _gpt_like(),
    "llama": lambda: _plain_like(),
    "llama_kvshare": lambda: _plain_like(kv_slots=[0, 1, 1, 1]),
    "llama_kvshare_win": lambda: _plain_like(window=8, kv_slots=[0, 1, 1, 1]),
}


@pytest.fixture(params=list(FLAVORS))
def config(request):
    return FLAVORS[request.param]()


@pytest.fixture
def manager():
    return ModelManager()


def build(manager, config, seed=0):
    return manager.create_model(config, device=torch.device("cpu"), seed=seed)
