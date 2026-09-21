"""
The materialized-tree config every modelcore.Model is built from -- the ONE format core
understands. See modelcore/docs/architecture.md's "The materialized config tree" section for the
design this continues.

Every component (embedding, block, unembedding, composer, shared component like RoPE) is one
ComponentSpec: its type under a "#type" key, its already-concrete constructor kwargs as siblings.
The "#" sigil can't collide with a parameter name (no Python identifier contains it), so no
namespacing wrapper is needed. A composer is a component like any other -- config.body's own
"#type" names the composer, and its per-layer block list is just one of its params (conventionally
named "blocks") -- so nothing in this schema privileges "a stack of blocks": a composer with
several block lists, or one nesting another composer, is expressible without a schema change.
"""

from __future__ import annotations

from dataclasses import dataclass, field

TYPE_KEY = "#type"
FORMAT_V1 = "modelcore.v1"
FORMAT = "modelcore.v2"  # stamped by to_dict(); from_dict dispatches on it (see ModelConfig.from_dict)
SUPPORTED_FORMATS = (FORMAT_V1, FORMAT)
COMMENT_PREFIX = "_"     # a key starting with this is a freeform comment, never a parameter
# What ModelConfig.template may say about how a model is talked to. Declarative only for now:
# "base" is plain completion; "nanochat" is the <|user_start|>... conversation format with
# <|python_start|>...<|python_end|> Python tool calls.
TEMPLATES = ("base", "nanochat")


def _split_comments(d: dict) -> tuple[dict, dict]:
    """(data, comments): `_`-prefixed keys are comments -- kept apart from the data so they can
    never reach a component constructor, and re-emitted by to_dict so they survive a round trip."""
    data = {k: v for k, v in d.items() if not k.startswith(COMMENT_PREFIX)}
    comments = {k: v for k, v in d.items() if k.startswith(COMMENT_PREFIX)}
    return data, comments


@dataclass
class AttentionLayerSpec:
    """Per-layer attention geometry: what the KV cache needs to allocate for this layer, and what
    the FLOPs/KV-bytes accounting in modelcore.stats needs to charge for it. window=-1 means
    unlimited/full context; a non-negative window is the number of preceding tokens attended to.
    -1 is the ONLY spelling of full attention: ModelConfig.sequence_len is just the maximum length
    a model was trained/allocated for, inference may run shorter, so a window equal to it would
    bake the training length into the architecture.
    kv_slot identifies which KVCache slot this layer reads/writes; None means "this layer owns a
    slot at its own position" (the default, one-slot-per-layer case). A layer that reuses an
    earlier layer's K/V (cross-layer KV sharing) sets kv_slot to that layer's slot instead."""
    n_head: int
    n_kv_head: int
    head_dim: int
    window: int = -1
    kv_slot: int | None = None


@dataclass
class ComponentSpec:
    """`comments` holds this spec's `_`-prefixed keys. They are deliberately not in `params`:
    params become constructor kwargs (and are checked against the constructor's signature), so a
    comment sitting there would be a validation error -- and, worse, indistinguishable from a
    mistyped parameter."""
    type: str
    params: dict = field(default_factory=dict)
    comments: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {TYPE_KEY: self.type, **self.comments, **{k: _unresolve(v) for k, v in self.params.items()}}

    @classmethod
    def from_dict(cls, d: dict) -> "ComponentSpec":
        d = dict(d)
        type_ = d.pop(TYPE_KEY)
        data, comments = _split_comments(d)
        return cls(type=type_, params={k: _resolve(v) for k, v in data.items()}, comments=comments)


@dataclass
class AdapterSpec:
    """One materialized low-rank adapter attached to a single Linear-valued target -- the same
    "already-concrete, never a derivation rule" convention ComponentSpec follows (see this
    module's docstring and modelcore/docs/architecture.md's "Adapters in the config tree").

    `target` is a module FQN relative to the Model root (e.g. "body.blocks.3.attn.c_q", exactly
    what Model.get_submodule resolves) -- concrete, never a pattern or a layer range; a host-side
    low-code layer (e.g. nanochat.architectures.adapters.expand_adapters) is what turns "every
    attn.c_q" into a list of these. `name` is the stable handle a caller enables/disables/re-adds
    an adapter by, and the key its delta's own params live under in the state dict (target +
    ".deltas." + name + ...) -- see modelcore.peft.apply.AdapterLinear. `type` names an entry in
    modelcore.peft.registry ("lora", "dora", ...); `params` are that delta class's own constructor
    kwargs (e.g. r/alpha/dropout for LoRA) besides in_features/out_features, which come from the
    target itself. `enabled` toggling is the whole point of this being config rather than a
    one-shot transform: flip it and reload, no retraining, no state-dict surgery.

    `comments` / `params_comments` hold `_`-prefixed keys at the spec's own level and inside its
    `params` respectively -- kept out of `params` for the same reason as ComponentSpec.comments."""
    target: str
    name: str
    type: str
    params: dict = field(default_factory=dict)
    enabled: bool = True
    comments: dict = field(default_factory=dict)
    params_comments: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {**self.comments, "target": self.target, "name": self.name, "type": self.type,
                "params": {**self.params_comments, **self.params}, "enabled": self.enabled}

    @classmethod
    def from_dict(cls, d: dict) -> "AdapterSpec":
        data, comments = _split_comments(d)
        params, params_comments = _split_comments(data["params"])
        return cls(
            target=data["target"], name=data["name"], type=data["type"],
            params=params, enabled=data["enabled"], comments=comments, params_comments=params_comments,
        )


def _resolve(value):
    """Recursively turn any dict carrying "#type" -- at any depth, including inside a list -- into
    a ComponentSpec. This one rule is what makes nested composers and multiple block lists work
    without the schema knowing about them in advance."""
    if isinstance(value, dict) and TYPE_KEY in value:
        return ComponentSpec.from_dict(value)
    if isinstance(value, list):
        return [_resolve(v) for v in value]
    return value


def _unresolve(value):
    if isinstance(value, ComponentSpec):
        return value.to_dict()
    if isinstance(value, list):
        return [_unresolve(v) for v in value]
    return value


def _count_blocks(spec) -> int:
    """Total number of block ComponentSpecs reachable under `spec`, however the composer(s)
    arrange them: sums every list bound to a "blocks" param, recursing into nested composers.
    Backs ModelConfig.n_layer below -- a convention (composers name their per-layer list
    "blocks"), not a schema requirement."""
    if not isinstance(spec, ComponentSpec):
        return 0
    total = 0
    for key, value in spec.params.items():
        if key == "blocks" and isinstance(value, list):
            total += len(value)
        elif isinstance(value, ComponentSpec):
            total += _count_blocks(value)
        elif isinstance(value, list):
            total += sum(_count_blocks(v) for v in value)
    return total


# Top-level keys a v2 dict may carry (besides "format" and `_` comments). Anything else is an error
# rather than being silently dropped -- a comment is *explicitly* a `_` key, so a stray key that
# isn't one is a mistake, and dropping it would also lose it on the next save.
_REQUIRED_KEYS = ("sequence_len", "vocab_size", "n_embd", "pad_vocab_size_to", "template",
                  "input", "body", "output")
_OPTIONAL_KEYS = ("reference", "shared", "adapters", "frozen", "meta", "tokenizer")


@dataclass(kw_only=True)
class ModelConfig:
    """A materialized architecture tree -- the only shape modelcore knows how to build. Nothing
    here has a default that changes the architecture: every value is stated, and a v2 dict that
    omits one is rejected (only modelcore.config.upgrade, converting a v1 dict, may supply one).

    `reference` optionally records how this config was produced (`{"preset": name, "kwargs":
    {...}}`, stamped by whatever depth-dial layer built it outside core) so a muP scaling-law
    reference model can be re-derived at a different depth without re-deriving the whole tree by
    hand, and so a display/tag name is available without core knowing about architecture names at
    all; a config with no `reference` (e.g. a from-scratch hand-written tree) has neither. It is
    machinery, not documentation -- `meta` is the free-form, never-interpreted counterpart.

    `template` says how the model is talked to (see TEMPLATES); declarative only for now.
    `tokenizer` is an opaque descriptor of the tokenizer the model expects (conventionally
    {"name", "fingerprint", "vocab_size", "special_tokens"}) -- modelcore knows nothing about
    tokenizers, so it is carried, never checked; a host reconciles it against `vocab_size`.
    `comments` holds top-level `_` keys.

    sequence_len is the maximum length the model was trained/allocated for, not a property of the
    architecture: full attention is window=-1 (see AttentionLayerSpec), never window=sequence_len.

    Only truly global, uniform-across-layers values live at this level: n_embd is the residual-
    stream width every block reads from and writes to, so it can't vary per layer without breaking
    the residual connection itself. Everything genuinely per-layer lives inside `body`."""
    sequence_len: int
    vocab_size: int
    n_embd: int
    pad_vocab_size_to: int
    template: str
    reference: dict | None = None
    shared: dict = field(default_factory=dict)   # name -> ComponentSpec, e.g. {"rope": ..., "norm": ...}
    input: ComponentSpec | None = None
    body: ComponentSpec | None = None
    output: ComponentSpec | None = None
    adapters: list = field(default_factory=list)  # list[AdapterSpec] -- see AdapterSpec above
    frozen: list = field(default_factory=list)     # module FQNs; that subtree gets
                                                    # requires_grad_(False) at build time (see
                                                    # Model.__init__). Concrete FQNs only, never a
                                                    # glob -- a host-side layer materializes these
                                                    # the same way it materializes `adapters`.
    meta: dict = field(default_factory=dict)        # free-form, host-owned, never interpreted
    tokenizer: dict | None = None                   # opaque descriptor -- see the docstring
    comments: dict = field(default_factory=dict)

    @property
    def padded_vocab_size(self) -> int:
        p = self.pad_vocab_size_to
        return ((self.vocab_size + p - 1) // p) * p

    @property
    def n_layer(self) -> int:
        """Total block count, however config.body arranges them -- a derived read-only property
        (not a dataclass field), since it can vary with tree content, not just depth."""
        return _count_blocks(self.body)

    def to_dict(self) -> dict:
        assert self.input is not None and self.body is not None and self.output is not None, (
            "ModelConfig.to_dict() requires input/body/output to already be set"
        )
        d = {
            "format": FORMAT, **self.comments,
            "sequence_len": self.sequence_len, "vocab_size": self.vocab_size, "n_embd": self.n_embd,
            "pad_vocab_size_to": self.pad_vocab_size_to, "template": self.template,
            "reference": self.reference,
            "shared": {k: v.to_dict() for k, v in self.shared.items()},
            "input": self.input.to_dict(), "body": self.body.to_dict(), "output": self.output.to_dict(),
        }
        # Optional annotations/extensions are omitted when empty rather than written as [] / {} /
        # null -- nothing in a model's architecture depends on them, so "absent" and "empty" mean
        # the same thing and a config that doesn't use them stays uncluttered.
        if self.adapters:
            d["adapters"] = [a.to_dict() for a in self.adapters]
        if self.frozen:
            d["frozen"] = list(self.frozen)
        if self.meta:
            d["meta"] = self.meta
        if self.tokenizer is not None:
            d["tokenizer"] = self.tokenizer
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        """Dispatches on d["format"]: "modelcore.v2" parses as-is; "modelcore.v1" -- or no "format"
        key at all, which is what a v1 dict looked like to this method before it read the key --
        goes through modelcore.config.upgrade first, the one place defaults are allowed; anything
        else is an error. Missing required keys and unknown non-comment keys are both errors, each
        reported with every offender named (not a bare KeyError on the first)."""
        d = dict(d)
        fmt = d.get("format", FORMAT_V1)
        if fmt not in SUPPORTED_FORMATS:
            raise ValueError(f"unsupported config format {fmt!r}; supported: {list(SUPPORTED_FORMATS)}")
        if fmt == FORMAT_V1:
            from modelcore.config.upgrade import upgrade_v1_to_v2
            d = upgrade_v1_to_v2(d)
        d.pop("format")
        data, comments = _split_comments(d)
        unknown = sorted(set(data) - set(_REQUIRED_KEYS) - set(_OPTIONAL_KEYS))
        if unknown:
            raise ValueError(
                f"unknown top-level key(s) {unknown} in a {FORMAT} config (a comment must start with "
                f"{COMMENT_PREFIX!r}); accepted: {sorted(set(_REQUIRED_KEYS) | set(_OPTIONAL_KEYS))}"
            )
        missing = [k for k in _REQUIRED_KEYS if k not in data]
        if missing:
            raise ValueError(f"a {FORMAT} config is missing required key(s) {missing} -- v2 has no defaults")
        return cls(
            sequence_len=data["sequence_len"], vocab_size=data["vocab_size"], n_embd=data["n_embd"],
            pad_vocab_size_to=data["pad_vocab_size_to"], template=data["template"],
            reference=data.get("reference"),
            shared={k: ComponentSpec.from_dict(v) for k, v in data.get("shared", {}).items()},
            input=ComponentSpec.from_dict(data["input"]),
            body=ComponentSpec.from_dict(data["body"]),
            output=ComponentSpec.from_dict(data["output"]),
            adapters=[AdapterSpec.from_dict(a) for a in data.get("adapters", [])],
            frozen=list(data.get("frozen", [])),
            meta=dict(data.get("meta", {})), tokenizer=data.get("tokenizer"), comments=comments,
        )


def resolve_reference_config(resolved_config: "ModelConfig", ref_depth: int, expand) -> "ModelConfig":
    """The muP scaling-law reference model (modelcore.scaling.derive_training_plan's d_ref) at
    ref_depth (12), for a config a caller's own depth-dial expander already resolved to
    `resolved_config`. Moved here from two identical copies (nanochat's
    architectures/presets.py, tinylab's presets.py) -- both only ever touched
    ModelConfig.reference, a modelcore-owned field, but need the caller's own preset registry to
    re-expand it, so `expand` (the caller's own `expand(preset_name, depth, **kwargs)`, e.g.
    nanochat.architectures.presets.expand) is a required parameter rather than baked in here --
    this module knows nothing about what presets exist. Every `expand()`-produced config always
    stamps a `reference` block on its own output (a caller convention, not enforced here), so this
    works uniformly for any preset."""
    assert resolved_config.reference is not None, (
        f"config has no 'reference' block, so its muP scaling-law reference model can't be "
        f"re-derived automatically at depth {ref_depth}"
    )
    return expand(resolved_config.reference["preset"], ref_depth, **resolved_config.reference["kwargs"])
