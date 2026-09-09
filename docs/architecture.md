# modelcore: architecture contract

`modelcore` is a standalone model subsystem: configs, architectures-as-data, and the machinery to
create/load/save models and optimizers and compute their stats. It has exactly one public
entrypoint, `ModelManager`, and a small set of value types that cross its boundary. It understands
exactly one config format — a materialized component tree — and nothing about a host application's
architecture *names*, CLI flags, checkpoint tags, or tokenizers. It has zero imports outside itself
(plus `torch`, and an optional `kernels` dependency for the FA3 kernel path) — see
`tests/test_standalone.py` for the mechanical proof.

In this repo, `nanochat/architectures/` is the layer that turns a `--depth` dial or an old
checkpoint into a tree for `modelcore` to build, and `nanochat/checkpoint_manager.py`/
`nanochat/engine.py` are the layer that adapts checkpoint naming/tokenizers/tool-use onto
`ModelManager`. See the repo root's `docs/architecture.md` for that side of the contract; this
document only covers `modelcore` itself.

```
modelcore/
├── manager.py         ModelManager -- the one entrypoint
├── model.py            Model -- the one model class, built from a config tree
├── generate.py         sample_next_token, generate_naive, Decoder (cached prefill+decode)
├── config/
│   ├── spec.py            ComponentSpec, ModelConfig, AttentionLayerSpec
│   └── validate.py        validate_config() -- structural + component-owned semantic checks
├── catalog.py           component registry: "#type" name -> (cls, needs, validate)
├── components/           linear, norm, rope, rotary, attention, mlp, block, embedding, unembedding
├── composers/             base, stack, backout
├── roles.py               parameter-role protocol (optimizer grouping)
├── stats.py               FLOPs/param/KV-bytes accounting, ModelStats
├── store.py               ArtifactStore protocol + FileSystemStore
├── runtime.py             Runtime (compute dtype, log sink) -- injected, not a global
├── precision/
│   └── fp8.py              Float8Linear + convert_to_float8_training (ModelManager.enable_fp8)
├── optim/                  MuonAdamW
├── kernels/                FA3/SDPA flash-attention interface
├── cache.py                KVCache
└── tests/                  modelcore's own test suite + goldens (see "Verifying" below)
```

## `ModelManager`: the one entrypoint

Everything a caller needs — create, load, save, or validate a model or its optimizer, measure a
config's cost, or drive generation — goes through one object:

```python
class ModelManager:
    # config
    def config_from_dict(self, d: dict) -> ModelConfig
    def config_to_dict(self, config: ModelConfig) -> dict
    def validate_config(self, config: ModelConfig) -> ValidationReport   # every error, not just the first

    # create
    def create_model(self, config, *, device, seed: int | None = None) -> Model
    def create_optimizer(self, model, hparams: OptimizerHparams | None = None) -> MuonAdamW

    # load / save
    def load_model(self, store, *, device, config=None, train: bool = False) -> Model
    def load_optimizer(self, model, store, *, rank=0, hparams=None) -> MuonAdamW | None
    def save_model(self, model, store) -> None
    def save_optimizer(self, optimizer, store, *, rank=0) -> None

    # stats & runtime helpers
    def stats(self, config: ModelConfig) -> ModelStats
    def new_kv_cache(self, model, *, batch_size, seq_len, device=None) -> KVCache
    def new_decoder(self, model, tokens, *, num_samples=1, max_tokens=None, device=None) -> Decoder

    # precision
    def enable_fp8(self, model, *, recipe="tensorwise", align=16, min_dim=128) -> Fp8Report
    def fp8_disabled(self, model)   # context manager; no-op if model has no fp8 modules
```

`create_model`/`load_model` own the meta-device dance (`torch.device("meta")` → `to_empty(device)`
→ `init_weights()`, and — for `load_model` — `load_state_dict(..., assign=True)` right after) so
no caller writes it themselves. Both call `validate_config` first and raise on any error — an
invalid tree never reaches `Model.__init__`.

`Model` itself (`modelcore/model.py`) is deliberately thin: `__call__(idx, targets=None,
kv_cache=None, loss_reduction=...)`, `.config`, `.get_device()`. It carries **no** accounting or
optimizer methods — no `layer_specs()`, `kv_cache_spec()`, `estimate_flops()`,
`num_scaling_params()`, `setup_optimizer()`. Those need a model only to read shapes/roles, which
`ModelManager.stats()`/`create_optimizer()` do from the outside. This is deliberate: a model
object answers "what do I compute", not "how much does that cost" or "how do I optimize myself".

## The materialized config tree

There is exactly one config shape `modelcore` understands (`modelcore/config/spec.py`):

```python
@dataclass
class ComponentSpec:
    type: str            # the catalog's "#type" name
    params: dict          # already-concrete constructor kwargs; may nest more ComponentSpecs

@dataclass
class ModelConfig:
    sequence_len: int
    vocab_size: int
    n_embd: int                    # the only truly global, uniform-across-layers value
    pad_vocab_size_to: int = 64
    reference: dict | None = None  # optional provenance: {"preset": name, "kwargs": {...}}
    shared: dict = {}              # name -> ComponentSpec, e.g. {"rope": ...}
    input: ComponentSpec | None = None    # embedding
    body: ComponentSpec | None = None     # the composer (a stack of blocks, or nested composers)
    output: ComponentSpec | None = None   # unembedding
```

Every component — embedding, block, unembedding, composer, a shared thing like RoPE — is one
`ComponentSpec`: its type under a `"#type"` key, already-concrete constructor kwargs as siblings.
A composer is a component like any other; its per-layer block list is just one of its own params
(conventionally `"blocks"`), so nothing in this schema privileges "a stack of blocks" — a composer
with several block lists, or one nesting another composer, needs no schema change.
`ModelConfig.n_layer` is a derived property (`_count_blocks` sums every `"blocks"`-named list,
recursing into nested composers), not a stored field, since it can vary with tree content.

`to_dict()`/`from_dict()` stamp/read a `"format": "modelcore.v1"` key. A dict with no `"format"`
key predates `modelcore` and needs the host application's own legacy migration first. There is no
`"arch"` field anywhere in this schema; `reference` (when present) is the closest equivalent, and
it's provenance, not something `modelcore` ever reads back to decide how to build the tree.

### Only concrete, already-decided values — no rules

A config tree carries no *derivation rules*, only their already-computed output. Every value that
used to be a rule lives outside `modelcore`, run once at tree-expansion time (in this repo, that's
`nanochat/architectures/derive.py`):

- `has_value_embed`: a plain `bool` a `gpt_block`'s params carry directly — not `None` meaning
  "derive the alternating-by-parity pattern from `n_layer`". `Block` (the class backing
  `"gpt_block"`) doesn't take `n_layer` at all, because it never needs to re-derive anything.
- `window`: a concrete int (or `-1` for full context) per block — not a pattern string like
  `"SSSL"` tiled at construction time.
- `kv_slot`/`produces_kv`: concrete per-block values — not a `kv_share_frac` float a component
  would need to interpret.

This is the fix for the abstraction leak the whole design guards against: a component like
`CausalSelfAttention` cannot have a method like `has_ve(layer_idx, n_layer)` (a policy about
*which* layers get a value embedding), because by the time `modelcore` ever sees a config, that
decision is already made. Materializing it is the depth-dial layer's job, not core's.

## Component contracts

Three module contracts (`modelcore/components/contracts.py`), each owning everything about its
own concern and nothing about how it's assembled into a model:

```python
class BaseEmbedding(nn.Module):
    def init_weights(self): ...
    def forward(self, idx, kv_cache=None): ...

class BaseBlock(nn.Module):
    def init_weights(self): ...
    def forward(self, x, x0, idx, kv_cache, kv_bus=None): ...
    def layer_spec(self): return None   # AttentionLayerSpec, or None if not attention-shaped

class BaseUnembedding(nn.Module):
    def init_weights(self): ...
    def forward(self, x, targets=None, loss_reduction="mean"): ...
```

Plus one more, for whatever owns `body` (`modelcore/composers/base.py`):

```python
class BaseComposer(nn.Module):
    def init_weights(self): ...
    def forward(self, x, idx, kv_cache): ...
    def layer_specs(self): ...   # list[AttentionLayerSpec], in forward-pass order
```

A component may know it must implement one of these contracts; it must not know anything about
*who* wires it in. Position encoding is deliberately not baked into `BaseBlock.forward`'s
signature (an architecture can swap RoPE for something else without touching the block/trunk
contract); `x0` (the post-embedding residual) is computed by whichever composer needs it
(`BackoutComposer`), not by the embedding.

### Every parameter needs a declared role

A module declares `PARAM_ROLES = {attr_name: role_string}` for parameters/submodules it directly
owns; a `Linear`'s `.weight` defaults to role `"matrix"` when undeclared; anything else undeclared
raises rather than silently defaulting into the wrong optimizer bucket. `collect_param_roles(module)`
(`modelcore/roles.py`) walks the tree; the escape hatch `param_roles(self) -> dict[str, list[Parameter]]`
lets a module (e.g. a tied `LMHead`) own its whole subtree's assignment in one place. This is also
why `modelcore.precision.fp8.Float8Linear` subclasses `Linear` rather than a bare `nn.Linear` — an
fp8-converted matmul weight still needs to resolve to role `"matrix"`.

`build_param_groups(role_params, policy)` turns `{role: [params]}` plus an **ordered**
`{role: hyperparameters}` policy into `MuonAdamW` param groups — the order is the on-disk
optimizer format (state is checkpointed and reloaded positionally by flat index across every
group). `ModelManager.create_optimizer` owns the one policy table every model in `modelcore` uses,
covering every role any cataloged component can produce (`build_param_groups` skips a policy role
with no params present, so a plain `llama`-shaped tree's optimizer groups look exactly like they
always did — nothing extra).

## The catalog

`modelcore/catalog.py` maps a `"#type"` name to `(cls, needs, validate)`. A component
self-registers at its own class definition:

```python
@register_component("gpt_block", needs=("n_embd", "padded_vocab_size", "rope", "runtime"),
                     validate=_validate_gpt_block)
class Block(BaseBlock):
    ...
```

`needs` names build-context values injected as constructor kwargs — derived globals
(`n_embd`, `vocab_size`, `padded_vocab_size`, `sequence_len`, `runtime`) that `Model.__init__`
computes once, plus `shared` components (built once and injected by name, e.g. `rope`) — so a
spec's `params` only ever needs to carry what's *not* derivable from context. `build_component`
resolves any nested `ComponentSpec` (or list of them) first, then calls `cls(**resolved_params,
**needed)`.

`modelcore/components/__init__.py` and `modelcore/composers/__init__.py` import every
component/composer module so its decorator runs; importing `modelcore` (or `modelcore.manager`)
imports both, so the catalog is always fully populated by the time a caller reaches
`ModelManager`.

### Validation

`validate_config(config)` never raises and never stops at the first problem — it returns a
`ValidationReport` with every error found, each anchored to a path (`body.blocks[3].n_kv_head`,
`shared.rope.head_dim`, `input.#type`):

- **Structural checks (core-owned, generic across any component)**: `input`/`body`/`output`
  present; every `#type` registered; every `needs` name available in the build context; no unknown
  or missing constructor params (checked via `inspect.signature`).
- **Semantic checks (component-owned, via the catalog's `validate` hook)**: `n_embd` divisible by
  `n_head`, a KV-sharing consumer (`produces_kv=False`) can't have `has_value_embed=True`,
  `produces_kv=False` requires an explicit `kv_slot` — things only the component knows the rule
  for. `modelcore/components/block.py`'s `_validate_gpt_block`/`_validate_plain_block` are the
  reference implementation.
- **Cross-layer checks (core-owned, but generic over any composer's block list, not hardcoded to
  one composer type)**: KV slots form a contiguous `0..M-1` range, a consumer's `kv_slot` points
  at an earlier producer, `n_kv_head`/`head_dim` are uniform across every attention-shaped block —
  computed structurally from block params, without building a real model.

`create_model`/`load_model`/`stats` all call `validate_config` first and raise `ValueError` on any
error, so an invalid tree is caught before a single tensor is allocated.

## `ModelStats` and accounting

`ModelManager.stats(config)` builds a model on `torch.device("meta")` (shapes/dtypes only, no real
weight values ever allocated — cheap regardless of model size) and returns a frozen snapshot:

```python
@dataclass(frozen=True)
class ModelStats:
    n_layer: int
    params_by_role: dict          # generic role -> numel, for every architecture uniformly
    num_params: int
    num_matmul_params: int
    layer_specs: list[AttentionLayerSpec]
    kv_cache_spec: dict            # what modelcore.cache.KVCache needs to allocate
    shape_summary: dict            # n_layer/n_embd/n_head/n_kv_head/sequence_len/window_pattern
    flops_per_token: int
    has_sliding_window: bool

    @property
    def num_scaling_params(self) -> int: ...   # matrix + unembedding roles (cleanest scaling laws)
    def decode_flops(self, context_len): ...
    def prefill_flops(self, num_tokens): ...
    def kv_bytes_per_token(self): ...
    def kv_read_bytes(self, context_len): ...
```

`shape_summary` reports a concrete value for `n_head`/`n_kv_head`/`window_pattern` when every
layer agrees, else the string `"mixed"` — one implementation for every tree, uniform or not,
rather than a separate "flat config" code path. `params_by_role` is the generic role-keyed dict
for *every* architecture — a host application wanting a different, legacy-shaped presentation
builds it from `params_by_role` at its own layer (in this repo, `scripts/base_train.py`'s
`_legacy_scaling_keys` does this for GPT's old six-key dict).

`AttentionLayerSpec.window = -1` means unlimited/full context; a non-negative int is the number of
preceding tokens attended to. `kv_slot = None` means "this layer owns a slot at its own position";
a layer that reuses an earlier layer's K/V sets it to that layer's slot instead — see "Cross-layer
KV sharing" below.

## `ArtifactStore`: how a model/optimizer gets its bytes

`modelcore/store.py` defines a narrow protocol — read/write a model state dict, an optimizer state
dict per rank, a config dict — and `FileSystemStore`, the default implementation: one checkpoint
directory + step (`model_{step:06d}.pt`, `meta_{step:06d}.json`'s `"model_config"` key,
`optim_{step:06d}_rank{N}.pt`). `write_config` merges into the meta.json's `"model_config"` key
rather than overwriting the file, since a host application typically writes its own sibling keys
(`val_bpb`, `user_config`, `tokenizer_fingerprint`, ...) into the same file — each side only ever
touches the key(s) it owns.

A store is deliberately narrow and duck-typed (not an ABC) — `ArtifactStore` is a protocol, not a
base class a caller is required to subclass. This is the seam a host application uses to adapt an
old, pre-`modelcore` format onto `ModelManager` without core ever learning that old formats exist:
in this repo, `nanochat.checkpoint_manager.LegacyCheckpointStore(FileSystemStore)` overrides
`read_config`/`read_model_state` to run a legacy checkpoint through `nanochat.architectures.legacy`
on first read, memoized, before handing bytes to `ModelManager.load_model` — `modelcore` itself
never has a legacy code path.

## `Runtime`: no more ambient globals

`modelcore/runtime.py`'s `Runtime` carries the values a component needs that aren't part of the
config: `compute_dtype` and a log sink. A component that needs it declares `needs=("runtime",)`,
same mechanism as `rope` or `n_embd`. `detect_compute_dtype()` reads `MODELCORE_DTYPE` from the
environment (CUDA capability, else fp32, as a fallback); a host application's own
`COMPUTE_DTYPE`-shaped global should source its value from `modelcore.runtime.DEFAULT_RUNTIME`
rather than the other way around, since `modelcore` has zero dependencies on its host (in this
repo, `nanochat.common.COMPUTE_DTYPE`/`COMPUTE_DTYPE_REASON` do exactly this, and also accept
`NANOCHAT_DTYPE` as a back-compat alias for `MODELCORE_DTYPE`).

## Adding a component, step by step

1. Write the `nn.Module`, obeying whichever contract it fits (`BaseEmbedding`/`BaseBlock`/
   `BaseUnembedding`/`BaseComposer`) — take explicit constructor kwargs, not a config object, so
   it doesn't assume any particular tree shape around it.
2. Declare `PARAM_ROLES` for every parameter it directly owns (or a `param_roles()` override); a
   `Linear`'s weight needs no declaration.
3. Register it: `@register_component("my_thing", needs=(...))`, naming exactly the build-context
   values its constructor needs beyond what's in the spec's own `params`. Add a `validate=`
   function if it has real constructor-level constraints worth catching before a model is built.
4. Import the module from `modelcore/components/__init__.py` (or `composers/__init__.py`) so the
   decorator runs.
5. If it needs testing at the tree level rather than in isolation, add a flavor to
   `modelcore/tests/conftest.py`'s `FLAVORS` dict. If it should be reachable from the host
   application's own depth-dial CLI, add it there too (in this repo, that's
   `nanochat/architectures/presets.py`).

## Cross-layer KV sharing

`AttentionLayerSpec.kv_slot` decouples layer index from KV-cache slot:
`kv_cache_spec()["num_kv_slots"]` (via `modelcore.stats.kv_cache_spec`) can be less than the layer
count when a layer's `kv_slot` points at an earlier layer's slot. `KVCache`'s constructor kwarg
and attribute are `num_kv_slots`/`n_slots`, and `get_slot_cache(slot)` returns that slot's
`(k_cache, v_cache)` view.

A layer built with `produces_kv=False` has no `c_k`/`c_v` at all; at forward time it reads an
earlier layer's already-RoPE'd/normed/scaled K/V out of a `kv_bus` dict (threaded through one
composer's forward pass) instead of computing its own — it only projects and rotates its own
queries (`RotaryEmbedding.apply_to_q`). Passing the producer's own K/V tensors back into
`flash_attn_with_kvcache` for the consumer (rather than `k=None`) sidesteps a real FA3-vs-SDPA
divergence in what `k=None` means: with a real cache, `k=None` tells FA3 "nothing new to insert
this call", so it reads exactly `cache_seqlens` cached tokens — correct for the producer, which
already wrote the shared slot earlier in the *same* forward pass, but wrong by exactly the new
token count for a consumer if it relied on `k=None` too.

`kv_cache.advance()` belongs to `Model.forward`, called once after the whole block/composer loop
runs — not to any one attention layer (a same-layer-count assumption breaks the moment a model has
fewer KV slots than layers).

## Intra-document masking

`doc_args` (a `modelcore.kernels.flash_attn.DocArgs`, built by `build_doc_args(idx, bos_token_id)`)
restricts attention to within each packed training row's own document, threaded through
`Model.forward` → the active composer → `BaseBlock.forward` → `CausalSelfAttention.forward`
alongside `kv_bus`, defaulting to `None` (today's behavior: attention sees the whole row) at every
hop. Training only — always `None` when `kv_cache is not None`, since one KV-cache row is one
document at decode time.

`build_doc_args` must be called **outside** any `torch.compile` region and its result passed in as
plain data: it derives boundaries via `nonzero()`, which recompiles every call inside a compiled
graph, and `flash_attn_varlen_func`'s `cu_seqlens` is padded to a fixed shape so the compiled
model's input shapes never change step to step (a real, measured cost otherwise — see
`docs/upstream/LOG.md`'s "Varlen Attention" entry: 25s/iter from a variable-shape `cu_seqlens`).

That fixed shape is `DEFAULT_MAX_DOCS_PER_ROW * batch_size` documents, **not** the true worst case
(every token its own document). This is load-bearing, not a style choice: the FA3 varlen kernel
sizes its backward-pass scratch off `cu_seqlens`'s declared length regardless of how many segments
are actually non-empty, so defaulting to the worst case (`batch_size * sequence_len`) made a real
2x H100 run's backward pass try to allocate 28GB of scratch for a declared batch of 131,072
sequences when the real batch had ~270 documents — an OOM against a model that otherwise fit in
70GB. `DEFAULT_MAX_DOCS_PER_ROW=64` is sized against ClimbMix's measured ~4.2 documents/row at
`sequence_len=2048`; `scripts/base_train.py --doc-masking-max-docs-per-row` overrides it for a
dataset/sequence-length combination that packs more.

Positions are **not** reset per document. RoPE attention scores depend only on the relative offset
`i − j` between two positions (see `modelcore/components/rope.py`'s note on this), and QK-norm
commutes with RoPE because a rotation preserves vector norm. With intra-document masking, every
surviving `(i, j)` pair already lies inside one document, so every relative offset a reset would
produce is identical to what the row's own absolute positions already give — resetting is a
bit-identical no-op that would trade a free `cos[:, T0:T0+T]` slice for a per-token gather, on
every layer, for nothing.

`Smear` (`modelcore/components/embedding.py`) is a known, deliberate gap: it blends each token's
embedding with its predecessor's across the whole row, upstream of q/k/v, where no attention mask
can reach — one token of leak per document boundary, in the `gpt` preset only (`llama*` presets
disable it).

## FP8 precision

`modelcore/precision/fp8.py` is a from-scratch, ~150-line tensorwise-dynamic-scaling FP8 training
path (drop-in for torchao's `Float8Linear`/`convert_to_float8_training` API, without the ~2000
lines torchao needs for rowwise scaling, FSDP float8 all-gather, and tensor-subclass dispatch —
see the module docstring for the full design rationale). `Float8Linear` subclasses
`modelcore.components.linear.Linear`, not a bare `nn.Linear` — this is what keeps an fp8-converted
weight resolving to role `"matrix"` under `collect_param_roles` and counted by
`modelcore.stats.num_matmul_params`; a subclass of the wrong base class fails both silently.

`ModelManager.enable_fp8(model, *, recipe="tensorwise", align=16, min_dim=128)` walks the tree
converting every eligible `Linear` (dims divisible by `align`, at least `min_dim` — below that,
quantization overhead dominates the matmul it's supposed to speed up) and returns an `Fp8Report`
(`num_linear`, `num_converted`, `num_skipped`). `ModelManager.fp8_disabled(model)` is a context
manager that temporarily swaps every `Float8Linear` back to a plain `Linear` sharing the same
weight/bias (for full-precision eval), restoring on exit; it's a no-op if the tree has no fp8
modules. Both are safe to call at any point in a model's lifecycle — an optimizer built after
`enable_fp8` sees ordinary `"matrix"`-role parameters, same as before conversion.

## Generation primitives

`modelcore/generate.py` holds the tokenizer-agnostic half of autoregressive generation:

- `sample_next_token(logits, rng, temperature=1.0, top_k=None)` — greedy at `temperature=0`,
  multinomial (optionally top-k-renormalized) otherwise.
- `generate_naive(model, tokens, max_tokens, ...)` — a slow, no-KV-cache reference implementation
  (recomputes the full forward pass every step); useful for checking a fast cached path against.
- `Decoder` — a batch-1 prefill of a prompt, replicated into an `num_samples`-row `KVCache`, then
  stepped one position at a time (`decoder.step(token_column) -> logits`). Reached via
  `ModelManager.new_decoder(model, tokens, *, num_samples=1, max_tokens=None, device=None)`.

None of this knows about tokenizers, special tokens, or tool use — a host application layers those
concerns on top, driving `Decoder` for the actual model-stepping (in this repo,
`nanochat.engine.Engine` adds the chat-token/calculator state machine around exactly this).

## The meta-device footgun

`Model.__init__` (and any component's `__init__`) may run under `torch.device("meta")` — shapes
and dtypes only, no real storage. `ModelManager.create_model`/`load_model` do this internally so a
caller never repeats the `to_empty(device)` → `init_weights()` dance. `RotaryEmbedding`'s
`cos`/`sin` buffers are `persistent=False` (never saved to a checkpoint) and only get real values
inside `init_weights()` — this is why `load_model` calls `init_weights()` even when *loading* a
checkpoint, right before `load_state_dict(..., assign=True)` overwrites everything else.

Each submodule owns its own `init_weights()` (called by its parent's, recursively) rather than one
function reaching into every submodule's internals by attribute path — this is what makes a
component usable inside a different tree without that tree needing to know the component's field
names. One consequence: the exact sequence of RNG calls during a from-scratch `init_weights()` is
not guaranteed stable across a refactor of shared code (same individual `torch.nn.init.*` calls,
potentially different order) — this does not affect *loading* an existing checkpoint (its saved
values fully override whatever `init_weights()` produced), only bit-for-bit reproducibility of a
brand-new from-scratch run at a given seed. This is why `modelcore/tests/goldens/tiny/*`'s
regression proof loads real saved weights into a freshly-built model rather than comparing two
independently-seeded `init_weights()` calls.

## Precision

Weight tensors stay fp32 (for optimizer precision); `modelcore.components.linear.Linear` casts to
`Runtime.compute_dtype` in `forward()`. Any new component should route its matmul weights through
`Linear` rather than a raw `nn.Linear`, both for this precision policy and so
`modelcore.stats.num_matmul_params` sees it — it's the structural marker that accounting scans
for, and the same marker `modelcore.precision.fp8` swaps in place of.

## Verifying a change is behavior-preserving

`modelcore/tests/goldens/*.json` (the `tiny_composed_*` set) record — for a seeded synthetic model
of every preset flavor — the state-dict fingerprint, every accounting number, and a forward-logits
hash, captured once and never expected to change. `modelcore/tests/test_manager.py`'s
`test_matches_pre_refactor_composed_golden` replays them.

```bash
python -m pytest modelcore/tests -v
```

runs the whole standalone suite, including `test_standalone.py` (an AST scan asserting zero
imports from a host application anywhere under `modelcore/`) and `test_precision.py`/
`test_generate.py` (fp8 role/accounting correctness, and `Decoder` vs `generate_naive` agreement).
The real proof of standalone-ness, occasionally worth re-running by hand:

```bash
cp -r modelcore /tmp/modelcore-check && cd /tmp/modelcore-check/.. \
  && PYTHONPATH=$(pwd) python -m pytest modelcore-check/tests -q
```

(rename the copy's parent-relative import to match, or simpler: copy to `<somewhere>/modelcore`
and run with that directory's parent on `PYTHONPATH`) — must pass with no host application on the
path at all.

A brand-new component or composer has no golden to diff against; verify it directly instead —
`manager.validate_config(config).ok`, `manager.create_optimizer(model)`'s groups partition
`model.parameters()` exactly (see `modelcore/tests/test_manager.py`'s generic suite, parametrized
over every flavor), and a forward/backward pass produces finite output and populates every
gradient.

For a change to a host application's own layer on top of `modelcore` (in this repo,
`nanochat/architectures/`, `nanochat/checkpoint_manager.py`, `nanochat/engine.py`), see the repo
root's `docs/architecture.md`.
