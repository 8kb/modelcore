# AGENTS.md

`modelcore` is a standalone model subsystem — configs, architectures-as-data, and the machinery to
create/load/save models and optimizers and compute their stats. Read
[docs/architecture.md](docs/architecture.md) for the full contract before touching anything here.
For the family-wide pattern this repo follows (one entrypoint, zero host imports, the tag-pin
consumption contract) see [llmllab/AGENTS.md](../llmllab/AGENTS.md) and
[llmllab/docs/subsystem-conventions.md](../llmllab/docs/subsystem-conventions.md).

A host application pins this repo by git tag (`pyproject.toml`'s `[tool.uv.sources]`) and consumes
it entirely through `ModelManager` — see [`llmllab/AGENTS.md`](../llmllab/AGENTS.md)'s family map
for which repos currently do that, and each one's own `docs/architecture.md` for its side of the
contract.

## Repo map

```
modelcore/
├── manager.py         ModelManager -- the one entrypoint
├── model.py            Model -- the one model class, built from a config tree
├── generate.py         sample_next_token, generate_naive, Decoder (cached prefill+decode)
├── evaluate.py           evaluate_bpb -- bits-per-byte (ModelManager.evaluate_bpb is the seam)
├── config/
│   ├── spec.py            ComponentSpec, ModelConfig, AttentionLayerSpec
│   ├── upgrade.py         upgrade_v1_to_v2 -- the ONLY place a default value may be supplied
│   └── validate.py        validate_config() -- structural + component-owned semantic checks
├── catalog.py           component registry: "#type" name -> (cls, needs, validate)
├── components/           linear, norm, rope, rotary, attention (incl. cross-layer KV sharing),
│                         mlp, block, embedding (+smear), unembedding
├── composers/             base, stack, backout
├── roles.py               parameter-role protocol (optimizer grouping)
├── stats.py               FLOPs/param/KV-bytes accounting, ModelStats
├── store.py               ArtifactStore protocol + FileSystemStore
├── runtime.py             Runtime (compute dtype, log sink) -- injected, not a global
├── precision/
│   └── fp8.py              Float8Linear + convert_to_float8_training (ModelManager.enable_fp8)
├── peft/                   AdapterLinear, LoRADelta/DoRADelta, apply_adapters/find_adapters/
│                           merge_adapters/strip_adapters -- see "Adapters are config" below
├── optim/                  MuonAdamW
├── kernels/                FA3/SDPA flash-attention interface
├── cache.py                KVCache
└── tests/                  this repo's own test suite
```

## Invariants that will bite you

- **`__init__` may run under `torch.device("meta")`.** `Model.__init__` (and any component's) must
  not compute anything that depends on real tensor *values* — only shapes/dtypes. Real
  initialization goes in `init_weights()`, called after `model.to_empty(device=...)`.
  `ModelManager.create_model`/`load_model` own this dance; nothing else should repeat it. See "The
  meta-device footgun" in [docs/architecture.md](docs/architecture.md).
- **No `torch.amp.autocast`.** Precision is `Runtime.compute_dtype`, injected into any component
  declaring `needs=("runtime",)` — not a bare global read off an attribute. Model weights stay
  fp32; `components/linear.py`'s `Linear` casts to `compute_dtype` in `forward()`. Route every
  matmul-participating parameter through it.
- **`Linear` is the structural marker for "matmul params".** `stats.num_matmul_params` finds every
  FLOPs-relevant parameter by scanning for `isinstance(m, Linear)`. A new matmul that uses a raw
  `nn.Linear` or bare `nn.Parameter` silently disappears from `ModelStats.flops_per_token`/
  `decode_flops`/`prefill_flops` and every FLOPs/s or MFU number derived from them. This is also
  why `precision/fp8.py`'s `Float8Linear` subclasses `Linear` rather than a bare `nn.Linear` — it
  used to subclass `nn.Linear` directly (pre-Stage-8, in nanochat's own `nanochat/fp8.py`), which
  meant an fp8-converted model's `collect_param_roles` raised outright (`Float8Linear.weight has no
  declared role`) the moment `ModelManager.create_optimizer` tried to build its param groups.
- **Every parameter needs a declared role.** `roles.collect_param_roles` walks the module tree and
  raises on any parameter it can't assign a role to (a `Linear.weight` defaults to `"matrix"`;
  anything else needs a `PARAM_ROLES` class attribute or a `param_roles()` override).
  `ModelManager.create_optimizer`/`ModelStats.params_by_role` are built on this, so a new
  `nn.Parameter` or submodule that forgets to declare a role raises at construction — far better
  than it silently defaulting into the wrong optimizer (e.g. Muon's shape-based matrix grouping).
  See [docs/architecture.md#component-contracts](docs/architecture.md#component-contracts).
- **A config tree carries only concrete, already-decided values, never a derivation rule — and a
  v2 tree has no defaults at all.** A default *is* a derivation rule ("if you don't say, it's 4x"),
  so `mlp`, `shared.norm`, `template`, `softcap`, `smear`, `over_compute`, `backout_lambda_init`,
  `kv_slot`/`produces_kv`, `pad_vocab_size_to` and an adapter's `enabled` are all required: omit one
  and the config is rejected, not completed. **Only `config/upgrade.py` (v1→v2) may supply a value**
  — its job is to write v1's implicit choices out so an old config builds the same model. Don't add
  a `=default` to a component constructor param that a config can set to save a line in a test.
  `has_value_embed` is a plain bool per block, `window` a concrete int, `kv_slot`/`produces_kv`
  concrete per-block values — never a pattern string or a fraction a component would need to
  interpret. Every rule that produces these values lives one layer up, in the host application's
  own preset/depth-dial layer (`nanochat/architectures/derive.py`, `tinylab/presets.py`), run once
  at tree-expansion time, outside this package entirely. A component asking "which layer am I" or
  "how many layers are there" to re-derive a policy is exactly the abstraction leak this package's
  design eliminated.
- **`window` is `-1` for full attention, never `sequence_len`.** `sequence_len` is the maximum a
  model was trained/allocated for; inference may run shorter, so a window equal to it bakes the
  training length into the architecture. The v1→v2 converter rewrites `window >= sequence_len` to
  `-1`; a host preset layer must emit `-1` in the first place.
- **A block's `head_dim` and `shared.rope`'s own `head_dim` are stated independently and never
  cross-checked.** `head_dim=null` on a block derives `n_embd // n_head` (what every config did
  before this param existed — the v1→v2 converter writes `null`, never a computed number); an
  explicit value decouples attention width from `n_head` entirely, e.g. more heads at a fixed
  `head_dim` that no longer equals `n_embd // n_head`. `_validate_kv_layout` requires uniform
  `head_dim` *across blocks* (what the KV cache needs), but nothing checks a block's `head_dim`
  against `shared.rope`'s — `apply_rotary_emb` just splits by `shared.rope`'s own `head_dim` and
  multiplies, so a mismatch is a runtime shape error (or a silently wrong broadcast on a
  same-size-coincidence), not a validation one. Keep them equal by hand.
- **A `_`-prefixed key is a comment and can never reach a constructor.** `ComponentSpec.comments` /
  `AdapterSpec.comments` / `ModelConfig.comments` hold them *beside* `params`, and `to_dict` writes
  them back, so they round-trip. Keep it that way: a comment in `params` would be a validation
  error and indistinguishable from a mistyped param. Conversely an unknown top-level key without
  the `_` is an error (`ModelConfig.from_dict`), never silently dropped.
- **`template`, `meta`, `tokenizer` are declarative/opaque.** `template` (`"base"`/`"nanochat"`) is
  validated against `TEMPLATES` and read by nothing yet; `meta` and `tokenizer` are carried, never
  interpreted or cross-checked — modelcore knows nothing about tokenizers, so a host reconciles
  `tokenizer` against `vocab_size` itself.
- **Norms are parameterless on purpose.** `rms_norm`/`layer_norm` have no learnable gain: a gain
  would need a `PARAM_ROLES` entry and would change the optimizer's positional on-disk layout, and
  a parameterless shared instance adds nothing to `state_dict()`, which is what keeps every existing
  checkpoint loadable. `"eps": null` on `rms_norm` is torch's own default (the input dtype's machine
  epsilon), and is what the converter writes for v1.
- **`ArtifactStore` is a real code path, not aspirational.** Saving/loading always goes through an
  `ArtifactStore` (`FileSystemStore` is the built-in one) — never `torch.save`/`torch.load`
  directly. A host application adapting an old on-disk format should subclass the store, not add a
  third way to read/write the same files (nanochat's `LegacyCheckpointStore` is a worked example —
  see its own [docs/architecture.md](https://github.com/8kb/nanochat/blob/master/docs/architecture.md)).
- **Optimizer state is checkpointed and reloaded positionally.** `torch.optim.Optimizer.state_dict()`
  flattens every parameter across every group into one global index order; a parameter that
  splits, merges, or moves group changes that indexing, and a same-size reorder corrupts state
  silently (no shape-mismatch error) rather than loudly. `ModelManager.create_optimizer`'s policy
  dict order is therefore part of the on-disk format, not just a style choice — see
  [docs/architecture.md#component-contracts](docs/architecture.md#component-contracts). A change
  that reorders or resplits needs a migration path in whichever host application has old shards to
  read, or they fail to load (nanochat's `nanochat/architectures/legacy.py` is a worked example).
- **`kv_cache.advance()` belongs to `Model.forward`, not the last attention layer.** It fires once,
  after the whole block/composer loop runs — broken the moment a model has fewer KV slots than
  layers (cross-layer KV sharing), since no layer's index then equals the slot count.
- **Multi-prompt decode is right-ragged, and `advance()` staying uniform is what keeps it correct.**
  `Decoder`/`generate_with_tools` accept `list[list[int]]`: each prompt is prefilled alone at batch
  1 and copied into its own row (`KVCache.prefill_row`), so row `i`'s KV occupies
  `[0, cache_seqlens[i])` — no left padding, no position offset, which is what FA3's
  `flash_attn_with_kvcache` natively means. Every row advances +1 per step, so the *differences*
  between rows never change; anything that stops stepping a row (compaction, per-row early exit)
  breaks that silently and has to replace `advance()`, not sit beside it. `KVCache.get_pos()`
  **raises** on a ragged cache (use `cache_seqlens`/`uniform_pos()`); `RotaryEmbedding._cos_sin`
  gathers per-row positions there, and the SDPA path masks with `col <= cache_seqlens[i] + t` (plus
  the window) — causal alone excludes every stale column, which is why `reset()` never zeroes KV.
  Rows that are all at one position take today's exact unmasked path, bit for bit. See
  `kernels.flash_attn._ragged_decode_mask`.
- **A freshly built model's logits ignore attention, RoPE, the window and the smear state.** Init
  zeroes every attention `c_proj`, every MLP down-projection and the smear `lambda_`, so logits are
  a function of the token embedding alone and a test of any of those passes whatever the code does
  (this hid a wrong-RoPE-position mutation until models were "awakened" — see `_build_awake` in
  `tests/test_generate.py`). Give such a test small random values in the zeroed parameters first.
- **Intra-document masking's `doc_args` must be built outside `torch.compile`.**
  `kernels.flash_attn.build_doc_args(idx, bos_token_id)` derives per-row document boundaries via
  `nonzero()` and must be called in the training loop, before `model(x, y, doc_args=...)` — never
  inside the compiled model itself. See
  [docs/architecture.md#intra-document-masking](docs/architecture.md#intra-document-masking) for
  why (a real, measured recompile cost otherwise) and why positions are deliberately not reset per
  document (RoPE + QK-norm make it a no-op).
- **`build_doc_args`'s `max_docs` default is a dataset-tuned guess, not a safe worst case.** It
  sizes the FA3 varlen kernel's backward-pass scratch allocation directly — defaulting it to
  `batch_size * sequence_len` (every token its own document) OOM'd a real 2x H100 run trying to
  allocate 28GB of scratch for a declared batch of 131,072 sequences when the real batch had ~270
  documents. `DEFAULT_MAX_DOCS_PER_ROW=64` is tuned against a real dataset's measured ~4.2
  documents/row at `sequence_len=2048`; a caller needs to override it for a different
  dataset/sequence-length combination rather than trust the default blindly.
- **`AttentionLayerSpec.kv_slot` decouples layer index from KV-cache slot.**
  `ModelStats.kv_cache_spec["num_kv_slots"]` can be `<= n_layer`: a layer whose `kv_slot` points at
  an earlier layer's slot (cross-layer KV sharing) shares that `KVCache` allocation instead of
  getting its own. `KVCache`'s constructor kwarg and attribute are `num_kv_slots`/`n_slots`, and
  `get_slot_cache(slot)` returns that slot's view — see "Cross-layer KV sharing" in
  [docs/architecture.md](docs/architecture.md) for the full mechanism, including a real
  FA3-vs-SDPA divergence in what `k=None` means to `flash_attn_with_kvcache` that a naive sharing
  implementation would hit.
- **Adapters (LoRA/DoRA) are config, not a transform.** Unlike `enable_fp8` (a one-shot call over
  an already-built model, invisible to `config.to_dict()`), `ModelConfig.adapters`/`.frozen` are
  fields on the config tree itself, applied automatically inside `Model.__init__` — the same
  "materialized DSL" `ComponentSpec` already is for architecture (see docs/architecture.md's
  "Adapters in the config tree"). This is *why* it's config: a caller can hand-edit a checkpoint's
  `meta.json` (enable/disable/add an `AdapterSpec`) and reload via `ModelManager.load_model`'s
  reconciling load, with no state-dict surgery. `frozen` is applied *before* adapters are attached
  (`Model.__init__`'s order, not incidental) — freezing shares the target's existing weight
  `Parameter` object into the new `AdapterLinear`, so the base stays frozen while a freshly
  constructed delta defaults to trainable; reversing the order would instead recurse into the
  delta too. An adapter's own on/off state (`AdapterLinear.set_enabled`) also toggles
  `requires_grad_` on the whole delta — a disabled delta never enters the forward computation, so
  leaving it `requires_grad=True` would put a permanently-`None`-grad parameter into an optimizer
  group, which `MuonAdamW.step()` dereferences unconditionally and crashes on.
- **`init_adapter_weights` is a separate pass from `init_weights()`, and must stay scoped to
  `AdapterLinear`.** `Model.init_weights()` calls it in its own pass, *after* every base weight
  already has real values (DoRA's magnitude is initialized from the base weight's own row norms).
  The pass is `isinstance(m, AdapterLinear)`, not "every module defining `init_adapter_weights`" —
  each delta *also* defines the method (so `AdapterLinear.init_adapter_weights` can call it on its
  own children with the right `base_weight` argument), and a second, duck-typed call directly on
  the delta would both waste RNG draws and hard-fail DoRA, whose signature requires `base_weight`.
- **New optimizer roles (`"adapter"`, `"adapter_scalar"`) are appended at the end of the policy
  dict**, same rule as every other role — see the optimizer-state-is-positional invariant below.
- **`build_param_groups` drops any `requires_grad=False` parameter**, not just leaves it ungrouped
  — a frozen or disabled-adapter parameter's `.grad` is always `None`, and `MuonAdamW.step()`
  dereferences it unconditionally.
- **`convert_to_float8_training`'s walk skips an `AdapterLinear`'s entire subtree** (checked
  *before* recursing, not after) — fp8-converting a LoRA/DoRA's own tiny `lora_A`/`lora_B` Linears,
  or the base weight underneath a delta, would silently drop the delta's contribution.
- **`ModelManager.evaluate_bpb`'s `token_bytes` is entirely caller-supplied.** modelcore knows
  nothing about tokenizers (see docs/architecture.md's tokenizer-free rule) -- `evaluate_bpb`
  accepts `token_bytes` as a plain vector (list, numpy array, or tensor) and converts it once,
  internally, with `torch.as_tensor`. A host typically gets it from its own data-prep layer (e.g.
  a datacore-prepared dataset's own `token_bytes()`); modelcore has no opinion on that.

## Testing

```bash
python -m pytest modelcore/tests -v
```

No GPU required for most of the suite — `test_optim.py` is module-level `skipif(not
cuda_available)`, and `TestFA3VsSDPA` in `test_kernels.py` needs an sm80/sm89/sm90 GPU for the real
FA3 kernel (the SDPA fallback classes in that file run fine on CPU); both skip cleanly on a
CUDA-less machine. `test_standalone.py` mechanically checks that nothing under `modelcore/` imports
a host application — see [docs/architecture.md](docs/architecture.md#verifying-a-change-is-behavior-preserving)
for the from-scratch standalone-copy recipe.

A change here that a host application depends on needs that host's own suite run against it too,
after an editable install (`uv pip install -e ../modelcore` from the host's venv) — see
[`llmllab/docs/subsystem-conventions.md`](../llmllab/docs/subsystem-conventions.md)'s tag-bump rule.
modelcore's own tests proving *this package* still works is necessary but not sufficient proof a
host is unaffected.

**Untested on a CUDA-less machine**, as a consequence of the above: the `bfloat16` compute path,
the real FA3 kernel path (vs. the SDPA fallback it's checked against), the real fp8 `_scaled_mm`
numerics (`precision/fp8.py` — the role/accounting bookkeeping around it is CPU-tested, see
`tests/test_precision.py`), and multi-GPU/DDP gradient reduction in `optim/`. Keep changes to those
paths conservative and prefer reasoning from the code plus the existing (CUDA-gated) tests over "I
ran it and it worked."

## Style

See [llmllab/AGENTS.md](../llmllab/AGENTS.md#style).
