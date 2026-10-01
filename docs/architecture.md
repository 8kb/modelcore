# modelcore: architecture contract

`modelcore` is a standalone model subsystem: configs, architectures-as-data, and the machinery to
create/load/save models and optimizers and compute their stats. It has exactly one public
entrypoint, `ModelManager`, and a small set of value types that cross its boundary. It understands
exactly one config format — a materialized component tree — and nothing about a host application's
architecture *names*, CLI flags, checkpoint tags, or tokenizers. It has zero imports outside itself
(plus `torch`, and an optional `kernels` dependency for the FA3 kernel path) — see
`tests/test_standalone.py` for the mechanical proof.

A host application's own preset/depth-dial layer is what turns a dial (or an old checkpoint) into a
tree for `modelcore` to build, and its own checkpoint-naming/tokenizer/tool-use layer is what adapts
that onto `ModelManager` — see that host's own `docs/architecture.md` for its side of the contract;
this document only covers `modelcore` itself.

```
modelcore/
├── manager.py         ModelManager -- the one entrypoint
├── model.py            Model -- the one model class, built from a config tree
├── generate.py         sample_next_token, Decoder (cached prefill+decode),
│                        generate_with_tools/ToolSpec/collect_batch (tool-use decode loop)
├── scaling.py            derive_training_plan -- muP horizon/batch-size/LR-scale derivation
├── config/
│   ├── spec.py            ComponentSpec, ModelConfig, AttentionLayerSpec
│   ├── upgrade.py         upgrade_v1_to_v2 (the only place a default value may be supplied),
│   │                      upgrade_v2_to_v3, remap_v2_name (v2 module FQN / state key -> v3)
│   └── validate.py        validate_config() -- structural + component-owned semantic checks
├── catalog.py           component registry: "#type" name -> (cls, needs, validate)
├── components/           linear, norm, rope, rotary, attention (the `attention` mixer), conv (the
│                         shared causal depthwise conv + the `short_conv` mixer), mamba2, mamba3, mlp, block (the one
│                         `block` class), features (output_gate, value_embed, resid_lambdas, canon,
│                         backout), embedding, unembedding
├── composers/             base, stack (the one body class)
├── convert.py             convert_checkpoint_v2_to_v3 + `python -m modelcore.convert`
├── roles.py               parameter-role protocol (optimizer grouping)
├── stats.py               FLOPs/param/KV-bytes accounting, ModelStats
├── store.py               ArtifactStore protocol + FileSystemStore (+ read_meta/update_meta,
│                        last_step)
├── runtime.py             Runtime (compute dtype, log sink) -- injected, not a global; also
│                        compute_init/compute_cleanup (device/DDP/seed bring-up) and
│                        peak_flops/peak_bandwidth (MFU/MBU hardware tables)
├── precision/
│   └── fp8.py              Float8Linear + convert_to_float8_training (ModelManager.enable_fp8)
├── optim/                  MuonAdamW, schedules.py (lr_multiplier/muon_momentum)
├── kernels/                FA3/SDPA flash-attention interface; ssm.py (SSD scan, mamba_ssm kernels)
├── cache.py                KVCache
└── tests/                  modelcore's own test suite (see "Verifying" below)
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
    def apply_schedule(self, optimizer, *, lr_mult=None, muon_momentum=None, muon_weight_decay=None) -> None

    # load / save
    def load_model(self, store, *, device, config=None, train: bool = False) -> Model
    def load_optimizer(self, model, store, *, rank=0, hparams=None) -> MuonAdamW | None
    def save_model(self, model, store) -> None
    def save_optimizer(self, optimizer, store, *, rank=0) -> None

    # stats & runtime helpers
    def stats(self, config: ModelConfig) -> ModelStats
    def new_kv_cache(self, model, *, batch_size, seq_len, device=None) -> KVCache
    def new_decoder(self, model, tokens, *, num_samples=1, max_tokens=None, device=None) -> Decoder

    # evaluate
    def evaluate_bpb(self, model, batches, steps, token_bytes, *, bos_token_id=None,
                      doc_masking_max_docs_per_row=None, padding_id=None) -> float

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
    comments: dict        # this spec's `_`-prefixed keys -- never in params

@dataclass(kw_only=True)
class ModelConfig:
    sequence_len: int              # the MAXIMUM length trained/allocated for -- not architecture
    vocab_size: int
    n_embd: int                    # the only truly global, uniform-across-layers value
    pad_vocab_size_to: int
    template: str                  # how the model is talked to: "base" | "chat_tools" (see below)
    reference: dict | None = None  # optional provenance: {"preset": name, "kwargs": {...}}
    shared: dict = {}              # name -> ComponentSpec, e.g. {"rope": ..., "norm": ...}
    input: ComponentSpec | None = None    # embedding
    body: ComponentSpec | None = None     # the composer (a stack of blocks, or nested composers)
    output: ComponentSpec | None = None   # unembedding
    meta: dict = {}                # free-form, host-owned, never interpreted
    tokenizer: dict | None = None  # opaque descriptor of the tokenizer the model expects
    comments: dict = {}            # top-level `_` keys
```

Every component — embedding, block, unembedding, composer, a shared thing like RoPE or the norm —
is one `ComponentSpec`: its type under a `"#type"` key, already-concrete constructor kwargs as
siblings. A composer is a component like any other; its per-layer block list is just one of its own
params (conventionally `"blocks"`), so nothing in this schema privileges "a stack of blocks" — a
composer with several block lists, or one nesting another composer, needs no schema change. A
block's feed-forward network is the same kind of thing: a nested `"mlp"` spec (see "Feed-forward and
norm are components" below). `ModelConfig.n_layer` is a derived property (`_count_blocks` sums every
`"blocks"`-named list, recursing into nested composers), not a stored field, since it can vary with
tree content.

### Format versions

`to_dict()` always stamps `"format": "modelcore.v3"`; `from_dict()` dispatches on it:

- `"modelcore.v3"` parses as-is.
- `"modelcore.v2"` goes through `modelcore.config.upgrade.upgrade_v2_to_v3` first.
- `"modelcore.v1"` — or **no `"format"` key at all**, which is what a v1 dict looks like to this
  method — goes through `upgrade_v1_to_v2` and then `upgrade_v2_to_v3`.
- Anything else raises `ValueError` naming the supported formats.

`upgrade_v1_to_v2` is **the only code in `modelcore` allowed to supply a default value**. A v1 dict
left a lot implicit — an MLP hardcoded per block type, a free-function norm, constructor defaults
like `softcap=15` — and the converter writes each of those out, so a v1 config keeps building the
exact same model (checked bit-for-bit against pre-change weights and logits). It also rewrites
`window >= sequence_len` to `-1`. A **v3** dict that omits any of them is rejected, not completed.
`upgrade_v2_to_v3` supplies no new architecture-affecting value: it restructures (see below) and
writes `null`/`true`/`[]` where a v2 block hardcoded them.

`ModelConfig.from_dict` also rejects, naming every offender: a missing required key
(`sequence_len`, `vocab_size`, `n_embd`, `pad_vocab_size_to`, `template`, `input`, `body`, `output`)
and any **unknown top-level key** — previously silently dropped, and then lost on the next save.
There is no `"arch"` field anywhere in this schema; `reference` (when present) is the closest
equivalent, and it's provenance, not something `modelcore` ever reads back to decide how to build
the tree.

### Comments

A key starting with `_` is a freeform comment, at any level — top level, any `ComponentSpec`
(including nested ones like a block's `mlp`), an adapter, and inside an adapter's `params`. They are
kept in a `comments` dict beside `params`, never in it, so they **can never reach a constructor** and
are never a validation error, yet **survive a round trip** (`to_dict` writes them back, first).
The `_` prefix is what makes the distinction explicit: a mistyped parameter (`"comment"`,
`"windw"`) is still an error, because it is not a comment.

### `template`, `meta`, `tokenizer`

- `template` (required; `"base"` or `"chat_tools"`) says how the model is talked to: `base` is plain
  completion, `chat_tools` is the `<|user_start|>…` conversation format with Python tool calls
  (`<|python_start|>…<|python_end|>`). The former name `"nanochat"` is a documented load-time alias
  (`TEMPLATE_ALIASES`): `from_dict` maps it, so older files keep loading and are re-saved with the new name. It is **declarative only for now** — `validate_config` checks
  it is one of `TEMPLATES` and nothing else reads it. Real validation (does the tokenizer carry the
  tokens the template needs?) is deliberately deferred.
- `meta` is a free-form dict for provenance (name, description, dates). Never interpreted. Distinct
  from `reference`, which is preset-re-expansion machinery.
- `tokenizer` is an opaque descriptor — conventionally `{"name", "fingerprint", "vocab_size",
  "special_tokens"}` — so a checkpoint can say which tokenizer it needs. `modelcore` knows nothing
  about tokenizers, so it is **carried, never checked**: a host reconciles it against `vocab_size`.

Both are omitted from `to_dict()` when empty/`None`, since nothing about the architecture depends on
them.

### Feed-forward and norm are components

A block's FFN is a nested spec, and the norm is a shared component, so neither is a hardcoded choice:

```json
{ "#type": "block", "layer_idx": 0, "mixer": { ... }, "features": [ ... ],
  "ffn": { "#type": "mlp", "activation": "relu2", "hidden_dim": 3072 } }
```

| `#type` | shape | `activation` |
|---|---|---|
| `mlp` | `c_proj(act(c_fc(x)))` | `relu2` \| `relu` \| `gelu` \| `silu` |
| `gated_mlp` | `down(act(gate(x)) * up(x))` | `silu` \| `gelu` \| `relu` |

`activation` and `hidden_dim` are both **required**: there is no `4 * n_embd` and no Llama rounding
rule here — those derivations belong to the host layer that materializes the tree. `shared.norm` is
`rms_norm` or `layer_norm` (`eps` required) and is injected into every component that needs it
(`needs=(..., "norm")`), including attention's QK-norm. Neither norm has a learnable gain, so a
shared instance adds nothing to `state_dict()` and every existing checkpoint loads unchanged.
`rms_norm`'s `"eps": null` means torch's own default — the input dtype's machine epsilon — which is
what every model trained before norm was configurable used, so it is a real value, not a placeholder.

### Blocks, mixers and features

There is **exactly one block class**: `Block` (`#type: "block"`), a pre-norm residual step
`x = x + mixer(norm(x)); x = x + ffn(norm(x))`. Everything that used to make one block type differ
from another is a component in a slot or a **feature**:

```json
{ "#type": "block", "layer_idx": 0,
  "mixer": { "#type": "attention", "n_head": 6, "n_kv_head": 6, "head_dim": null, "window": -1,
             "kv_slot": null, "produces_kv": true,
             "features": [ {"#type": "value_embed", "gate_channels": 12},
                           {"#type": "output_gate", "granularity": "block", "block_size": 8, "in_channels": null} ] },
  "ffn": { "#type": "mlp", "activation": "relu2", "hidden_dim": 3072 },
  "features": [ {"#type": "resid_lambdas", "resid_lambda_init": 1.15, "x0_lambda_init": 0.2} ] }
```

"nanogpt" and "plain" are just two feature sets on this class. **nanogpt** is the modified GPT of
karpathy/nanochat (the family's reference architecture): `resid_lambdas` (per-layer residual and
x0 mixing scalars) on the block, `value_embed` (a token-indexed value embedding, gated, on
alternating layers) on the attention, `smear` on the input embedding (a cheap bigram-like mix of the
previous token), `backout` on the stack (subtracts a mid-depth residual from the final one), a
squared-ReLU FFN, and sliding-window "SSSL" attention. **plain** is a standard pre-norm stack:
no features, a gated SiLU FFN. A gated variant of either adds an `output_gate` on the attention.
These are host preset names, not modelcore concepts; the tree only ever contains the concrete
components. `gpt_block`/`plain_block`/`gated_*_block` and the `backout` composer
exist only as v2 names that `upgrade_v2_to_v3` rewrites — there is no class behind them. There is no
new block type, now or later: a new token mixer (Mamba, convolutions) is a new **mixer**, a new
trick is a new **feature**.

**Mixer** (`BaseMixer`): whatever fills the block's `mixer` slot. `forward(x, idx, cache, bus,
doc_args)`, `layer_spec()` (an `AttentionLayerSpec`, or `None` if it holds no KV cache),
`init_weights()`, and `bind_layer(layer_idx)` (called by the owning block, so `kv_slot: null` can
mean "my own slot at my layer_idx" without a mixer taking `layer_idx` as a param). In a model where
not every layer is attention, `layer_idx` is not a contiguous slot number, so such a config states
every `kv_slot` explicitly.

**Feature** (`BaseFeature`): a registered component that declares `HOOKS`. A host (`Block`,
`Attention`, `StackComposer`) declares the `HOOK_POINTS` it exposes and calls every
feature at each of them (`FeatureHost`); every hook is `hook(value, *context) -> value`, so several
features on one point chain in list order.

| host | hook | signature | features today |
|---|---|---|---|
| `block` | `residual_in` | `(x, x0) -> x` | `resid_lambdas` |
| `block` | `pre_mixer`, `pre_ffn` | `(h, cache, doc_args) -> h`, on the normed mixer / ffn input | `canon` |
| `attention` | `values` | `(v, x, idx) -> v` | `value_embed` |
| `attention` | `output` | `(y, x) -> y`, `y` is `(B, T, H, D)` | `output_gate` |
| `stack` | `after_block`, `finish` | `(x, i, state) -> x`, `(x, state) -> x` | `backout` |

Validation rejects a feature on a host whose `HOOK_POINTS` don't cover its `HOOKS`, a type listed
twice, and anything in `features` that isn't a `BaseFeature`. Adding a hook point is a code change
on the host, never a format change. Features live in a `ModuleDict` keyed by `#type`, so state-dict
keys don't depend on list order: `body.blocks.0.mixer.features.output_gate.proj.weight`,
`body.blocks.3.features.resid_lambdas.resid_lambda`, `body.features.backout.backout_lambda`. `stack`
always threads `x0` (the post-embedding activations) to every block, whether or not a feature
reads it.

`output_gate` is Qwen "Gated Attention" (G1 placement): after SDPA and before `c_proj`,
`y = y * sigmoid(W_g · x[..., :in_channels])`, where `x` is the block's normed mixer input.

| `granularity` | gates | `block_size` |
|---|---|---|
| `head` | `n_head` | `null` |
| `element` | `n_head * head_dim` | `null` |
| `block` | `n_head * head_dim / block_size` (contiguous outputs inside a head) | positive int dividing `head_dim` |

`in_channels: null` means all `n_embd` residual channels feed the gate (same "null = derive"
convention as `head_dim`); otherwise the first `in_channels`. `block_size` is required in every
spec. The gate projection is a `Linear` (role `matrix`, counted in matmul FLOPs). `value_embed` is
the ResFormer-style value residual, `v += 3*sigmoid(gate(x[..., :gate_channels])) * embed(idx)`
(`embed` is role `value_embedding`, `gate` a `Linear`); a KV-sharing consumer layer
(`produces_kv=false`) has no V of its own and can't carry it.

### Recurrent mixers and per-row state

The mixer slot is polymorphic. `attention` holds a KV cache and reports an `AttentionLayerSpec`; a
mixer with a fixed-size state instead reports a `RecurrentLayerSpec(kind, state_elems,
fwd_flops_per_token)`; `Model.layer_specs()` has one entry per block (attention spec, recurrent
spec, or `None`) and everything in `modelcore.stats` filters on the type. Consequences:

- **Cache.** A recurrent mixer keeps its state in `KVCache.state` (the dict `smear` already uses),
  keyed by layer (`short_conv.<layer_idx>`, `canon.<layer_idx>.<site>`), any rank as long as the
  first dim is the batch: `prefill` expands it to every row, `prefill_row` copies it into one row.
  A model with no attention at all has `num_kv_slots=0` (`kv_cache_spec` returns zeros; the k/v
  tensors are empty). Ragged decode needs nothing extra: each prompt is prefilled alone at batch 1.
- **Stats.** `ModelStats.state_elems_per_row`/`state_bytes_per_row()` is the constant per-row
  state (recurrent mixers + stateful features), the counterpart of `kv_bytes_per_token()`, which
  is 0 for a pure-conv model. `extra_fwd_flops_per_token` (conv taps, later scans) enters
  `flops_per_token` at 3x (forward + backward), `decode_flops` at 1x whatever the context, and
  `prefill_flops` per token; matmuls stay counted structurally through `Linear`.
- **Intra-document masking.** `doc_args.doc_ids` (built by `build_doc_args`) is honoured by every
  layer kind: a conv tap only reads a token of the same document, so a packed row equals running
  each document alone. Tested per mixer and per model.
- **KV slots in a hybrid.** `kv_slot: null` means `layer_idx`, which is not a contiguous slot number
  once some layers aren't attention, so a hybrid config states every attention layer's `kv_slot`
  explicitly (0..M-1 over the attention layers only).
- **Optimizer.** Depthwise conv filters have role `conv` (AdamW, no weight decay,
  `OptimizerHparams.conv_lr`), appended last in the policy table.

`short_conv` is the gated short convolution (LFM2-style): `b, c, v = in_proj(x); y = b * v`, a
causal depthwise conv over `y` (`kernel_size` taps, `(n_embd, kernel_size)` filter), then
`out_proj(c * conv(y))`. It needs no positional encoding, no RoPE in `shared`, and no KV cache; its
state is the last `kernel_size - 1` values of `b * v`. `canon` (Physics of Language Models 4.1) is a
block feature, not a mixer: `h = h + causalconv(h)` on the normed mixer input and/or ffn input
(`sites`), so it composes with any mixer. Both use `components/conv.py`'s one
`causal_depthwise_conv(x, weight, state, doc_ids)`, so full-sequence, prefill and single-step
decode are the same arithmetic.

### Mamba-2

`mamba2` is the Mamba-2 mixer (`components/mamba2.py`): `in_proj` → `[z, x, B, C, dt]`, a depthwise
causal conv (with bias) over `[x, B, C]` then SiLU, `dt = softplus(dt + dt_bias)`, `A = -exp(A_log)`,
the SSD scan, `norm(y * silu(z))`, `out_proj`. Every param is required and concrete: `d_state`,
`head_dim`, `expand` (`d_inner = expand * n_embd`, `n_head = d_inner / head_dim`), `n_groups` (B/C
shared across the heads of a group), `kernel_size`, `chunk_size`, and the reference's init dials
`dt_min`/`dt_max`/`dt_init_floor`/`A_init_min`/`A_init_max`. Deviations from the reference: the gated
norm is the shared parameterless one (no learnable gain, per the norms rule), and `out_proj` is zero
initialised. Roles: `in_proj`/`out_proj` matrix (Muon), conv filter and bias `conv`, and the per-head
`A_log`/`dt_bias`/`D` role `ssm` (AdamW, no weight decay, `OptimizerHparams.ssm_lr`). It needs no
positional encoding (a pure-Mamba config carries no `rope`), no KV cache, and keeps per row the conv's
last `kernel_size-1` inputs plus an `(n_head, head_dim, d_state)` state in `KVCache.state`.

The scan is `modelcore/kernels/ssm.py`: `ssd_scan` (chunked, any length, initial state in / final
state out) and `ssd_step` (one decode token). The reference is pure PyTorch and is what runs off
CUDA; on CUDA it loads `kernels-community/mamba-ssm` (`mamba_chunk_scan_combined`,
`selective_state_update`) through the `kernels` hub, like flash_attn.py loads FA3, and falls back
with the reason recorded. Document resets are exact in the reference (masks on the within-chunk
decay matrix, on which tokens reach a chunk's state, and on the carry between chunks -- not a
`-inf` decay, whose cumulative sums cancel catastrophically) and are passed to the kernel as
`seq_idx`. The reference is checked against a sequential recurrence (chunk sizes from 1 to beyond
the sequence, boundaries inside a chunk, at its edge, several per chunk); the kernel path against the
reference only on a GPU (`tests/test_ssm.py::TestKernelVsReference`), so treat it as unverified until
that has run.

### Mamba-3 (SISO)

`mamba3` (`components/mamba3.py`) follows the reference (`state-spaces/mamba`'s `modules/mamba3.py`
and `ops/triton/mamba3/mamba3_siso_step.py`). Against Mamba-2:

- **Exponential-trapezoidal discretization.** `h_t = a_t h_{t-1} + b_t k_{t-1} x_{t-1} + g_t k_t x_t`
  with `a = exp(A_t dt_t)`, `g = lam dt`, `b = (1 - lam) dt a`, `lam = sigmoid(...)` per head and
  token, and a **data-dependent** `A_t = -max(heavy_tail(dd_A), A_floor)`. The state sees the
  previous token too, an implicit width-2 convolution: **there is no short conv** in the layer.
- **Complex state as a data-dependent rotation of B and C.** Both are rotated by the same
  cumulative angle `sum_i tanh(w_i) pi dt_i` (per head, per pair; `w` is a projection shared by
  the heads, `dt` is per head), so only the relative rotation between tokens survives in
  `C_t . B_s`. **Mamba-3 needs no positional RoPE** (`shared.rope`): the recurrence orders tokens,
  and the angles come from the input, not the position. The rotation reuses
  `components/rope.py`'s `apply_rotary_emb` on the first `2 * num_rope_angles` dims (`rope_fraction`
  0.5 or 1.0; the reference pairs neighbours where this repo's RoPE pairs halves -- the same
  rotation up to a fixed permutation of dims). Attention layers in a hybrid keep `shared.rope`;
  making them NoPE would be a later feature.
- **BC norm and B/C biases.** The shared parameterless norm on B and C (the reference's carries a
  learnable gain), then a learnable per-head bias on each, initialised to 1.
- No `A_log`, no conv. Params/roles: `in_proj`/`out_proj` matrix; `dt_bias`, `B_bias`, `C_bias`, `D`
  role `ssm`. The reference's optional output-projection norm is not implemented. `mimo_rank` is
  in the spec from day one (`B_bias`/`C_bias` are `(n_head, mimo_rank, d_state)`); the validator
  rejects `mimo_rank > 1` until MIMO is implemented.

**Scan.** No new kernel: with `c_t = dt_t (1 - lam_t)` and `s_t = h_t + c_{t+1} k_t x_t`, the state
`s` obeys a plain scan `s_t = a_t s_{t-1} + (g_t + c_{t+1}) k_t x_t`, and
`y_t = q_t . s_t - c_{t+1} (q_t . k_t) x_t`. `c_{t+1}` is 0 at the last token of a call and of a
document, so the scan's final state is the true `h` and a packed row equals its documents. The scan
is `kernels.ssm.ssd_scan_decay` (decay and input weight given directly); on CUDA it goes through
`mamba_chunk_scan_combined` with `A = -1`, `dt = -a`, `x / dt`. The reference repo also ships
dedicated Mamba-3 Triton/CuTe kernels; those are not used here (not in the hub build, and
unverifiable without a GPU). Streaming keeps `h`, the previous token's rotated `k` and `x`, and
the running angle in the cache and continues from `s_{-1} = h + c_0 k_{-1} x_{-1}`; one-token decode
is the direct recurrence, and tests check it against the scan and against the paper's recurrence
written out token by token with explicit 2x2 rotation blocks.

### Migrating from v2

`upgrade_v2_to_v3` (pure dict surgery) rewrites `gpt_block`/`plain_block`/`gated_*` to `block`
(`n_head`/`n_kv_head`/`head_dim`/`window`/`kv_slot`/`produces_kv` move into a nested `attention`
mixer, `mlp` becomes `ffn`, `has_value_embed` becomes a `value_embed` feature, `attn_gate` an
`output_gate`, the lambdas a `resid_lambdas`), and the `backout` composer to a `stack` with a
`backout` feature. It also rewrites adapter targets and `frozen` FQNs (`remap_v2_name`), so an old
config carries its adapters forward.

Weights are a **separate, explicit step**: `modelcore.convert.convert_checkpoint_v2_to_v3(src_store,
dst_store)` (or `python -m modelcore.convert SRC_DIR DST_DIR [--step N]`) renames every state-dict key
(`attn.` → `mixer.`, `mlp.` → `ffn.`, `attn.gate.` → `mixer.features.output_gate.`,
`attn.value_embed`/`ve_gate` → `mixer.features.value_embed.embed`/`.gate`, `resid_lambda`/`x0_lambda`
→ `features.resid_lambdas.*`, `body.backout_lambda` → `body.features.backout.backout_lambda`) and
writes the upgraded config. `ModelManager.load_model` on a pre-v3 checkpoint raises an error pointing
at the converter. **Optimizer state is not converted** — it is positional and v3 reaches parameters
in a different order — so a converted checkpoint can't resume training.

A `ComponentSpec` tree built in code from v2 component types is not carried forward: `to_dict()`
stamps v3 and `validate_config` rejects the v2 type names. Only a v2 *dict* (a file) is upgraded.
Anything that reads tree internals (a block's `params["window"]`, an adapter FQN like
`body.blocks.0.attn.c_q`) sees the v3 shape after the upgrade.

### Only concrete, already-decided values — no rules

A config tree carries no *derivation rules*, only their already-computed output. Every value that
used to be a rule lives outside `modelcore`, run once at tree-expansion time, in the host
application's own preset/depth-dial layer:

- a `value_embed` feature present or absent per layer — not a `None` meaning "derive the
  alternating-by-parity pattern from `n_layer`". `Block` doesn't take `n_layer` at all, because it
  never needs to re-derive anything.
- `window`: a concrete int per block, or **`-1` for full context** — never a pattern string like
  `"SSSL"`, and never `sequence_len` as a spelling of "full". `sequence_len` is only the maximum a
  model was trained/allocated for; inference may run shorter, so a window equal to it would bake the
  training length into the architecture.
- `ffn`: a nested spec stating the activation and inner width — not a `4 * n_embd` a component would
  compute for itself.
- `kv_slot`/`produces_kv`: concrete per-block values — not a `kv_share_frac` float a component
  would need to interpret.

The same rule is why a v3 tree has **no defaults at all**: a default *is* a derivation rule ("if you
don't say, it's 4x"). Omit `ffn`, `features` (an empty list is a value), `shared.norm`, `template`, or any formerly-defaulted constructor
param (`softcap`, `smear`, `over_compute`, `backout_lambda_init`, `kv_slot`, `produces_kv`,
`pad_vocab_size_to`, an adapter's `enabled`) and the config is rejected. Only the upgraders above
write those values in.

This is the fix for the abstraction leak the whole design guards against: a component like
`Attention` cannot have a method like `has_ve(layer_idx, n_layer)` (a policy about
*which* layers get a value embedding), because by the time `modelcore` ever sees a config, that
decision is already made. Materializing it is the depth-dial layer's job, not core's.

## Component contracts

Module contracts (`modelcore/components/contracts.py`), each owning everything about its
own concern and nothing about how it's assembled into a model:

```python
class BaseEmbedding(nn.Module):
    def init_weights(self): ...
    def forward(self, idx, kv_cache=None): ...

class BaseBlock(nn.Module):
    def init_weights(self): ...
    def forward(self, x, x0, idx, kv_cache, kv_bus=None, doc_args=None): ...
    def layer_spec(self): return None   # AttentionLayerSpec, or None if it holds no KV cache

class BaseMixer(nn.Module):             # the block's token-mixing slot (attention, ...)
    def init_weights(self): ...
    def bind_layer(self, layer_idx): ...
    def forward(self, x, idx, cache, bus=None, doc_args=None): ...
    def layer_spec(self): return None

class BaseFeature(nn.Module):           # a trick in a host's `features` list
    HOOKS = ()                          # hook-point names it implements, one method each
    def bind(self, host): ...
    def init_weights(self): ...

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
(`StackComposer`), not by the embedding.

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
@register_component("block", needs=("norm",), validate=_validate_block)
class Block(BaseBlock, FeatureHost):
    ...
```

`needs` names build-context values injected as constructor kwargs — derived globals
(`n_embd`, `vocab_size`, `padded_vocab_size`, `sequence_len`, `runtime`) that `Model.__init__`
computes once, plus `shared` components (built once and injected by name, e.g. `rope`, `norm`) — so a
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
  `n_head`, a KV-sharing consumer (`produces_kv=False`) can't carry a `value_embed` feature,
  `produces_kv=False` requires an explicit `kv_slot` — things only the component knows the rule
  for. `modelcore/components/attention.py`'s `_validate_attention` is the reference implementation.
- **Feature checks (core-owned)**: a host's `features` holds only registered `BaseFeature`s whose
  `HOOKS` the host's `HOOK_POINTS` cover, each `#type` at most once; a block's `mixer` must be a
  `BaseMixer`.
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
    layer_specs: list              # per block: AttentionLayerSpec | RecurrentLayerSpec | None
    kv_cache_spec: dict            # what modelcore.cache.KVCache needs to allocate
    shape_summary: dict            # n_layer/n_embd/n_head/n_kv_head/sequence_len/window
    flops_per_token: int
    has_sliding_window: bool
    state_elems_per_row: int       # recurrent mixers' + stateful features' cache elements per row
    extra_fwd_flops_per_token: int # non-matmul, non-attention forward FLOPs (conv taps, scans)

    @property
    def num_scaling_params(self) -> int: ...   # matrix + unembedding roles (cleanest scaling laws)
    def decode_flops(self, context_len): ...
    def prefill_flops(self, num_tokens): ...
    def kv_bytes_per_token(self): ...
    def kv_read_bytes(self, context_len): ...
    def state_bytes_per_row(self): ...
```

`shape_summary` reports a concrete value for `n_head`/`n_kv_head`/`window` when every
layer agrees, else the string `"mixed"` — one implementation for every tree, uniform or not,
rather than a separate "flat config" code path. `params_by_role` is the generic role-keyed dict
for *every* architecture — a host application wanting a different, legacy-shaped presentation
builds it from `params_by_role` at its own layer (in this repo, `scripts/base_train.py`'s
`_legacy_scaling_keys` does this for GPT's old six-key dict).

`AttentionLayerSpec.window = -1` means unlimited/full context — the only spelling of it; a
non-negative int is the number of preceding tokens attended to. `kv_slot = None` means "this layer owns a slot at its own position";
a layer that reuses an earlier layer's K/V sets it to that layer's slot instead — see "Cross-layer
KV sharing" below.

## `ArtifactStore`: how a model/optimizer gets its bytes

`modelcore/store.py` defines a narrow protocol — read/write a model state dict, an optimizer state
dict per rank, a config dict — and `FileSystemStore`, the default implementation: one checkpoint
directory + step (`model_{step:06d}.pt`, `meta_{step:06d}.json`'s `"model_config"` key,
`optim_{step:06d}_rank{N}.pt`). `write_config` merges into the meta.json's `"model_config"` key
rather than overwriting the file, since a host application typically writes its own sibling keys
(`val_bpb`, `user_config`, `tokenizer_fingerprint`, ...) into the same file — each side only ever
touches the key(s) it owns. `FileSystemStore.read_meta()`/`update_meta(dict)` give a host the same
read-merge-write for *its* sibling keys (never touching `"model_config"`), and the module-level
`last_step(checkpoint_dir)` finds the highest saved step — both are format mechanics a host used to
reimplement per app; naming/tag policy (which directory, auto-discovery across tags) stays with
the host.

A store is deliberately narrow and duck-typed (not an ABC) — `ArtifactStore` is a protocol, not a
base class a caller is required to subclass. This is the seam a host application uses to adapt an
old, pre-`modelcore` format onto `ModelManager` without core ever learning that old formats exist —
`modelcore` itself never has a legacy code path.

## `Runtime`: no more ambient globals

`modelcore/runtime.py`'s `Runtime` carries the values a component needs that aren't part of the
config: `compute_dtype` and a log sink. A component that needs it declares `needs=("runtime",)`,
same mechanism as `rope` or `n_embd`. `detect_compute_dtype()` reads `MODELCORE_DTYPE` from the
environment (CUDA capability, else fp32, as a fallback); a host application's own
`COMPUTE_DTYPE`-shaped global should source its value from `modelcore.runtime.DEFAULT_RUNTIME`
rather than the other way around, since `modelcore` has zero dependencies on its host.

The same module also holds `compute_init(device_type="cuda", *, seed=42, backend="nccl", log=None)`
/`compute_cleanup()` (device/seed/DDP bring-up: seeds torch, sets tf32 matmul precision on CUDA,
inits `torch.distributed` when torchrun's env is present) and `peak_flops(device_name)`/
`peak_bandwidth(device_name)` (hardcoded per-GPU tables, the MFU/MBU denominators) — not part of
the injected-`Runtime`-value contract above (no component ever needs a device or a peak-flops
number), but living here because this is where a host's own runtime-bringup module already lives.
`compute_init` keeps the 5-tuple return shape `(is_ddp, ddp_rank, ddp_local_rank, ddp_world_size,
device)` both hosts already unpack, unchanged from before this moved.

## Scaling-law horizon derivation

`modelcore/scaling.py`'s `derive_training_plan(...)` is pure math (no I/O, no model) turning a
model's scalar stats (`ModelStats.num_scaling_params`, `.flops_per_token`) and its depth-12 muP
reference model's `num_scaling_params` into a `TrainingPlan`: the training horizon (iterations,
resolved from an explicit request, a FLOPs budget, or a data:param ratio, in that precedence
order), an auto-derived batch size (Power Lines' `B ~ D^0.383` scaling, clamped to a power of 2)
when the caller didn't pin one, and the LR/weight-decay corrections that follow from the batch size
actually chosen. `B_REF` (the empirically-measured optimal batch size at d12) is a module constant
but also a `b_ref=` keyword — an empirical measurement, not a derived law, so a caller who
re-measures it for their own setup isn't stuck with the shipped value. A host's own preset/depth-
dial layer is what supplies every input; this module never builds a model or reads a config tree.

`modelcore/optim/schedules.py`'s `lr_multiplier`/`muon_momentum` are the per-step schedule shapes
(linear warmup → hold → linear warmdown; a momentum ramp with its own warmup window) both hosts
used as their default — kept as plain functions of the step index, easy to replace with a
different shape entirely. `ModelManager.apply_schedule` is the part that isn't swappable per-host:
it mutates a `MuonAdamW`'s live `param_groups` (`"lr"` from `"initial_lr"`, and — for `"muon"`-kind
groups only — `"momentum"`/`"weight_decay"`), which is the same on-disk optimizer format
`create_optimizer` builds (see "Every parameter needs a declared role" above) — every value passed
to it is `None`-by-default (a no-op), so a caller can drive only the parts it wants scheduled.

`modelcore/generate.py` holds the tokenizer-agnostic half of autoregressive generation:

- `sample_next_token(logits, rng, temperature=1.0, top_k=None)` — greedy at `temperature=0`,
  multinomial (optionally top-k-renormalized) otherwise.
- `Decoder` — a batch-1 prefill of a prompt, replicated into an `num_samples`-row `KVCache`, then
  stepped one position at a time (`decoder.step(token_column) -> logits`). Reached via
  `ModelManager.new_decoder(model, tokens, *, num_samples=1, max_tokens=None, device=None)`.
- `generate_with_tools(model, manager, tokens, *, num_samples=1, max_tokens=None, temperature=1.0,
  top_k=None, seed=42, terminal_ids, tools=())` — the tool-use decode loop built on `Decoder`:
  forces a row to stop on any id in `terminal_ids`, and recognizes each `ToolSpec(start_id, end_id,
  result_start_id, result_end_id, run)` in `tools`, calling `run(captured_token_ids)` when a tool's
  `end_id` is seen and force-injecting the result (wrapped in `result_start_id`/`result_end_id`)
  when `run` returns tokens (`None` means "inject nothing" — wrong tool usage isn't fatal). Yields
  `(token_column, token_masks)` per step, same shape as `Decoder`-driven code always produced by
  hand. `collect_batch(stream, terminal_ids, prompt_tokens, num_samples)` drains such a stream into
  `(results, masks)` — each a list of `num_samples` token-id lists, terminal tokens excluded.

None of this knows about tokenizers or special-token *names* — a host resolves its own special
tokens to ids and, for tools, supplies what a captured expression actually evaluates to (`run`);
a host's engine is a thin adapter over exactly this loop, owning its own tool logic (e.g. an
`eval()` sandbox) as the one `ToolSpec.run` callback.

## Bits-per-byte evaluation

`modelcore/evaluate.py`'s `evaluate_bpb` (public seam: `ModelManager.evaluate_bpb`) reports loss as
bits-per-byte rather than the usual mean loss — a vocab-size-independent metric, so a checkpoint
trained against a different tokenizer's vocab is still comparable. Instead of averaging loss over
tokens, it sums loss and sums *bytes* (the target tokens' UTF-8 byte lengths) independently and
divides, over `steps` batches drawn from a plain iterable of `(x, y, ...)` (any extra elements
after the first two, e.g. a resumable loader's cursor state, are ignored).

`token_bytes` is the one tokenizer-shaped input, and it stays a caller concern, same rule as
generation's tokenizer-free split above: it's accepted as a plain vector (list, numpy array, or
tensor) and converted once with `torch.as_tensor`, so modelcore never has to know how a host
obtained it. Its contract: length `vocab_size`, value = that token's byte count, `0` for any token
to exclude from the metric (a special token like `<|bos|>`, or padding) — `0` doubles as the
exclusion mask, so a token with a real length is counted and one without isn't, in the same pass.
A target of `-1` (`ignore_index`) is excluded regardless of its byte length.

`bos_token_id`/`doc_masking_max_docs_per_row`/`padding_id` (all optional) forward straight to
`build_doc_args`, restricting attention to within each packed row's own document — matching
whatever masking training used keeps val bpb comparable to the training loss it's evaluating. When
`dist.is_initialized()` and `world_size > 1`, the byte and loss sums are `all_reduce`d before
dividing, so every rank reports the identical, fully-reduced value.

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
brand-new from-scratch run at a given seed. This is why a host application's own regression
goldens should load real saved weights into a freshly-built model rather than comparing two
independently-seeded `init_weights()` calls (a host carrying such goldens keeps them in its own repo).

## Precision

Weight tensors stay fp32 (for optimizer precision); `modelcore.components.linear.Linear` casts its
weight to the input's dtype (`x.dtype`, which is `Runtime.compute_dtype` after the embedding) in `forward()`. Any new component should route its matmul weights through
`Linear` rather than a raw `nn.Linear`, both for this precision policy and so
`modelcore.stats.num_matmul_params` sees it — it's the structural marker that accounting scans
for, and the same marker `modelcore.precision.fp8` swaps in place of.

## Constants that are part of the named architecture

Some numbers are not config keys on purpose (YAGNI: a constant becomes a key only when an
experiment needs it, and changing one is a new architecture, not a tuning knob): the smear gate
width (24 channels), the QK scale (1.2 on each of Q and K, after QK-norm), the RoPE base (100000),
the init stds (embedding 0.8, unembedding 0.001, value-embed/gate inits) and the zero-init of lambdas.
`softcap` is the exception: a `lm_head` takes `null` for "no softcap" (validated: `null` or a
positive number). `DEFAULT_MAX_DOCS_PER_ROW` (flash-attention varlen scratch sizing) is tuned to
the datasets it was measured on; pass `max_docs` to `build_doc_args` for a denser packing.

## Verifying a change is behavior-preserving

`modelcore`'s own net is its parametrized suite (`modelcore/tests/test_manager.py`, run over every
flavor in `conftest.py`'s `FLAVORS` dict): validation, forward/backward finiteness, param-role
partitioning, optimizer-group partitioning, layer-spec/KV-cache-spec consistency, seed
reproducibility, and save/load round trips. The one golden set here, `modelcore/tests/goldens/` (used by `test_v3_migration.py`), holds v2
checkpoints whose logits must stay bit-identical through the v2→v3 upgrade and converter. Goldens
proving a *host's* behavior across a refactor live in that host, not here.

```bash
python -m pytest modelcore/tests -v
```

runs the whole suite, including `test_standalone.py` (an AST scan asserting zero imports from a
host application anywhere under `modelcore/`) and `test_precision.py`/`test_generate.py` (fp8
role/accounting correctness, and `Decoder` vs a naive no-cache decode agreement). The real proof of
standalone-ness, occasionally worth re-running by hand from this repo's root:

```bash
cp -r modelcore /tmp/modelcore-check && cd /tmp/modelcore-check/.. \
  && PYTHONPATH=$(pwd) python -m pytest modelcore-check/tests -q
```

(copy the top-level `modelcore` directory — the one containing both the `modelcore/` package and
its `tests/` — to `<somewhere>/modelcore` and run with that directory's parent on `PYTHONPATH`) —
must pass with no host application on the path at all.

A brand-new component or composer has no fixture to diff against; verify it directly instead —
`manager.validate_config(config).ok`, `manager.create_optimizer(model)`'s groups partition
`model.parameters()` exactly (see `modelcore/tests/test_manager.py`'s generic suite, parametrized
over every flavor), and a forward/backward pass produces finite output and populates every
gradient.

**Behavior preservation for a host application's use of `modelcore` is that host's own concern, not
this repo's.** A change here that a host depends on should be checked against that host's own
suite too, after an editable install (`uv pip install -e ../modelcore` from its venv) — passing
modelcore's own suite is necessary but not proof a host is unaffected. See
[`llmllab/docs/subsystem-conventions.md`](../llmllab/docs/subsystem-conventions.md)'s tag-bump rule
and its [golden-placement section](../llmllab/docs/subsystem-conventions.md#where-a-golden-belongs)
for the general contract.
