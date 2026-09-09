# modelcore

A standalone model subsystem: configs, architectures-as-data, and the machinery to create, load,
save, and validate transformer-shaped models and their optimizers, and compute their FLOPs/param/
KV-cache-bytes stats. Zero dependency on any host application — only `torch` (plus an optional
`kernels` install for the FA3 attention kernel path; falls back to SDPA without it).

It has exactly one public entrypoint, `ModelManager`, and understands exactly one config format —
a materialized component tree (`ModelConfig`/`ComponentSpec`). It knows nothing about architecture
*names*, depth dials, CLI flags, checkpoint tags, or tokenizers: that's a host application's job,
sitting on top and producing an already-concrete tree for `modelcore` to build.

See [docs/architecture.md](docs/architecture.md) for the full contract.

## Quickstart

```python
import torch
from modelcore import ComponentSpec, ModelConfig, ModelManager

config = ModelConfig(
    sequence_len=1024, vocab_size=32768, n_embd=768,
    shared={"rope": ComponentSpec("rotary", {"head_dim": 64})},
    input=ComponentSpec("token_embedding", {"smear": False}),
    body=ComponentSpec("stack", {"blocks": [
        ComponentSpec("plain_block", {"layer_idx": i, "n_head": 12, "n_kv_head": 12, "window": -1})
        for i in range(12)
    ]}),
    output=ComponentSpec("lm_head", {}),
)

manager = ModelManager()
report = manager.validate_config(config)
assert report.ok, report.errors

model = manager.create_model(config, device=torch.device("cpu"), seed=0)
stats = manager.stats(config)
print(stats.num_params, stats.flops_per_token, stats.kv_cache_spec)

optimizer = manager.create_optimizer(model)

idx = torch.randint(0, config.vocab_size, (2, 16))
logits = model(idx)
```

Saving/loading goes through an `ArtifactStore` — `FileSystemStore` is the built-in one (a
directory + step convention):

```python
from modelcore import FileSystemStore

store = FileSystemStore("/tmp/my_checkpoint", step=0)
manager.save_model(model, store)
manager.save_optimizer(optimizer, store)

reloaded = manager.load_model(store, device=torch.device("cpu"))
```

## Tests

```bash
python -m pytest modelcore/tests -v
```

No GPU required — CUDA-only tests (the real `_scaled_mm`/FA3 numerics) skip automatically.
`modelcore/tests/test_standalone.py` mechanically checks that nothing under `modelcore/` imports
a host application; see [docs/architecture.md](docs/architecture.md#verifying-a-change-is-behavior-preserving)
for the full verification recipe, including a from-scratch standalone-copy check.
