"""
Shared config-tree builders for modelcore's test suite. Builds trees directly with
ComponentSpec/ModelConfig -- no dependency on nanochat.architectures (that package exists to
*produce* such a tree from a depth dial or an old checkpoint; modelcore's own tests only ever
consume an already-materialized one). The first four flavors mirror the four presets
nanochat.architectures.presets.expand() knows how to derive, so a bug that trips one of these
generic tests would trip the corresponding preset too.
"""
import pytest
import torch

from modelcore import AdapterSpec, ComponentSpec, ModelConfig, ModelManager


RMS_NORM = lambda: ComponentSpec("rms_norm", {"eps": None})


def _mixer(n_head, n_kv_head, head_dim, window, kv_slot=None, produces_kv=True, features=()):
    return ComponentSpec("attention", {"n_head": n_head, "n_kv_head": n_kv_head, "head_dim": head_dim,
                                       "window": window, "kv_slot": kv_slot, "produces_kv": produces_kv,
                                       "features": [f() for f in features]})


def _gpt_like(n_layer=4, n_head=2, n_kv_head=2, n_embd=64, head_dim=32, vocab_size=128, sequence_len=32, window=-1,
              mlp=None, norm=None, attn_gate=None):
    # A v3 tree states everything: an mlp per block, a shared norm, a template, no constructor
    # defaults. `mlp`/`norm` are overridable so a flavor can exercise a non-default choice.
    # "gpt" is a feature set on the one Block class: resid_lambdas on the block, value_embed (on
    # alternating layers) on the attention, backout on the stack.
    mlp = mlp or (lambda: ComponentSpec("mlp", {"activation": "relu2", "hidden_dim": 4 * n_embd}))
    blocks = []
    for i in range(n_layer):
        mixer_features = []
        if i % 2 == (n_layer - 1) % 2:
            mixer_features.append(lambda: ComponentSpec("value_embed", {"gate_channels": 12}))
        if attn_gate is not None:
            mixer_features.append(attn_gate)
        blocks.append(ComponentSpec("block", {
            "layer_idx": i,
            "mixer": _mixer(n_head, n_kv_head, head_dim, window, features=mixer_features),
            "ffn": mlp(),
            "features": [ComponentSpec("resid_lambdas", {
                "resid_lambda_init": 1.15 - 0.10 * i / max(n_layer - 1, 1),
                "x0_lambda_init": 0.20 - 0.15 * i / max(n_layer - 1, 1)})],
        }))
    return ModelConfig(
        sequence_len=sequence_len, vocab_size=vocab_size, n_embd=n_embd, pad_vocab_size_to=64, template="base",
        shared={"rope": ComponentSpec("rotary", {"head_dim": head_dim, "over_compute": 10}),
                "norm": (norm or RMS_NORM)()},
        input=ComponentSpec("token_embedding", {"smear": True}),
        body=ComponentSpec("stack", {"blocks": blocks, "features": [
            ComponentSpec("backout", {"backout_layer": n_layer // 2, "backout_lambda_init": 0.2})]}),
        output=ComponentSpec("lm_head", {"softcap": 15}),
    )


def _plain_like(n_layer=4, n_head=2, n_kv_head=2, n_embd=64, head_dim=32, vocab_size=128, sequence_len=32,
                 window=-1, kv_slots=None, mlp=None, norm=None, attn_gate=None):
    # Llama-style FFN width: 2/3 of 4x, rounded up to a multiple of 256 -- computed here, by the
    # test's own "host layer", not by modelcore. "llama" is the same Block class with no features.
    hidden = 256 * ((int(2 * (4 * n_embd) / 3) + 255) // 256)
    mlp = mlp or (lambda: ComponentSpec("gated_mlp", {"activation": "silu", "hidden_dim": hidden}))
    blocks = []
    for i in range(n_layer):
        kv_slot = None if kv_slots is None else kv_slots[i]
        produces_kv = True if kv_slots is None else kv_slots[i] == i
        blocks.append(ComponentSpec("block", {
            "layer_idx": i,
            "mixer": _mixer(n_head, n_kv_head, head_dim, window, kv_slot, produces_kv,
                            features=[] if attn_gate is None else [attn_gate]),
            "ffn": mlp(), "features": [],
        }))
    return ModelConfig(
        sequence_len=sequence_len, vocab_size=vocab_size, n_embd=n_embd, pad_vocab_size_to=64, template="base",
        shared={"rope": ComponentSpec("rotary", {"head_dim": head_dim, "over_compute": 10}),
                "norm": (norm or RMS_NORM)()},
        input=ComponentSpec("token_embedding", {"smear": False}),
        body=ComponentSpec("stack", {"blocks": blocks, "features": []}),
        output=ComponentSpec("lm_head", {"softcap": 15}),
    )


def _gpt_lora():
    """_gpt_like() with the whole body frozen and a LoRA adapter on two attention projections in
    layer 0 -- runs apply_adapters/the freeze-then-apply ordering, the "adapter" optimizer role,
    and the reconciling load_model path through every generic test in test_manager.py. Embedding/
    unembedding/shared stay trainable, so the frozen-vs-trainable split is meaningfully exercised
    rather than degenerating to "everything is frozen" or "nothing is"."""
    config = _gpt_like()
    config.frozen = ["body"]
    config.adapters = [
        AdapterSpec(target="body.blocks.0.mixer.c_q", name="t0", type="lora", params={"r": 4, "alpha": 8}),
        AdapterSpec(target="body.blocks.0.mixer.c_v", name="t0", type="lora", params={"r": 4, "alpha": 8}),
    ]
    return config


def _gpt_gated_mlp():
    """gpt-style blocks with a gated GELU FFN at an unrounded, non-4x width -- proves the mlp slot is
    genuinely free, not just a re-spelling of the two hardcoded shapes."""
    return _gpt_like(mlp=lambda: ComponentSpec("gated_mlp", {"activation": "gelu", "hidden_dim": 100}))


def _llama_layer_norm():
    """llama-style blocks with a plain (ungated) SiLU FFN of odd width, under a layer_norm shared norm."""
    return _plain_like(mlp=lambda: ComponentSpec("mlp", {"activation": "silu", "hidden_dim": 90}),
                       norm=lambda: ComponentSpec("layer_norm", {"eps": 1e-5}))


def _gpt_decoupled_head_dim():
    """head_dim stated explicitly and NOT equal to n_embd // n_head: n_head=8 * head_dim=16 = 128,
    double n_embd=64. Proves attention width is genuinely decoupled from n_embd (c_q/c_k/c_v/c_proj
    size off n_head*head_dim, not off n_embd) -- exercised through every generic parametrized test
    (forward, backward, stats, save/load, optimizer groups, kv_cache_spec) for free."""
    return _gpt_like(n_head=8, n_kv_head=8, head_dim=16)


def _gate(granularity, block_size=None, in_channels=None):
    return lambda: ComponentSpec("output_gate", {"granularity": granularity, "block_size": block_size,
                                               "in_channels": in_channels})


FLAVORS = {
    "gpt": lambda: _gpt_like(),
    "llama": lambda: _plain_like(),
    "llama_kvshare": lambda: _plain_like(kv_slots=[0, 1, 1, 1]),
    "llama_kvshare_win": lambda: _plain_like(window=8, kv_slots=[0, 1, 1, 1]),
    "gpt_lora": _gpt_lora,
    "gpt_gated_mlp": _gpt_gated_mlp,
    "llama_layer_norm": _llama_layer_norm,
    "gpt_decoupled_head_dim": _gpt_decoupled_head_dim,
    "gpt_gated_head": lambda: _gpt_like(attn_gate=_gate("head")),
    "llama_gated_element": lambda: _plain_like(attn_gate=_gate("element", in_channels=16)),
    "llama_kvshare_gated_block": lambda: _plain_like(kv_slots=[0, 1, 1, 1], attn_gate=_gate("block", 8)),
}


@pytest.fixture(params=list(FLAVORS))
def config(request):
    return FLAVORS[request.param]()


@pytest.fixture
def manager():
    return ModelManager()


def build(manager, config, seed=0):
    return manager.create_model(config, device=torch.device("cpu"), seed=seed)
