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
FORMAT = "modelcore.v1"  # stamped by to_dict(); a dict with no "format" key predates modelcore
                          # entirely and needs the host application's own legacy migration first.


@dataclass
class AttentionLayerSpec:
    """Per-layer attention geometry: what the KV cache needs to allocate for this layer, and what
    the FLOPs/KV-bytes accounting in modelcore.stats needs to charge for it. window=-1 means
    unlimited/full context; a non-negative window is the number of preceding tokens attended to.
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
    type: str
    params: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {TYPE_KEY: self.type, **{k: _unresolve(v) for k, v in self.params.items()}}

    @classmethod
    def from_dict(cls, d: dict) -> "ComponentSpec":
        d = dict(d)
        type_ = d.pop(TYPE_KEY)
        return cls(type=type_, params={k: _resolve(v) for k, v in d.items()})


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
    one-shot transform: flip it and reload, no retraining, no state-dict surgery."""
    target: str
    name: str
    type: str
    params: dict = field(default_factory=dict)
    enabled: bool = True

    def to_dict(self) -> dict:
        return {"target": self.target, "name": self.name, "type": self.type,
                "params": dict(self.params), "enabled": self.enabled}

    @classmethod
    def from_dict(cls, d: dict) -> "AdapterSpec":
        return cls(
            target=d["target"], name=d["name"], type=d["type"],
            params=dict(d.get("params", {})), enabled=d.get("enabled", True),
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


@dataclass
class ModelConfig:
    """A materialized architecture tree -- the only shape modelcore knows how to build. `reference`
    optionally records how this config was produced (`{"preset": name, "kwargs": {...}}`, stamped
    by whatever depth-dial layer built it outside core) so a muP scaling-law
    reference model can be re-derived at a different depth without re-deriving the whole tree by
    hand, and so a display/tag name is available without core knowing about architecture names at
    all; a config with no `reference` (e.g. a from-scratch hand-written tree) has neither.

    Only truly global, uniform-across-layers values live at this level: n_embd is the residual-
    stream width every block reads from and writes to, so it can't vary per layer without breaking
    the residual connection itself. Everything genuinely per-layer lives inside `body`."""
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_embd: int = 768
    pad_vocab_size_to: int = 64
    reference: dict | None = None
    shared: dict = field(default_factory=dict)   # name -> ComponentSpec, e.g. {"rope": ...}
    input: ComponentSpec | None = None
    body: ComponentSpec | None = None
    output: ComponentSpec | None = None
    adapters: list = field(default_factory=list)  # list[AdapterSpec] -- see AdapterSpec above
    frozen: list = field(default_factory=list)     # module FQNs; that subtree gets
                                                    # requires_grad_(False) at build time (see
                                                    # Model.__init__). Concrete FQNs only, never a
                                                    # glob -- a host-side layer materializes these
                                                    # the same way it materializes `adapters`.

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
            "format": FORMAT,
            "sequence_len": self.sequence_len, "vocab_size": self.vocab_size, "n_embd": self.n_embd,
            "pad_vocab_size_to": self.pad_vocab_size_to, "reference": self.reference,
            "shared": {k: v.to_dict() for k, v in self.shared.items()},
            "input": self.input.to_dict(), "body": self.body.to_dict(), "output": self.output.to_dict(),
        }
        # Omitted when empty (rather than an explicit []) so a pre-adapters config serializes
        # byte-identically to before -- every existing golden/checkpoint dict must round-trip
        # unchanged.
        if self.adapters:
            d["adapters"] = [a.to_dict() for a in self.adapters]
        if self.frozen:
            d["frozen"] = list(self.frozen)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        d = dict(d)
        d.pop("format", None)
        return cls(
            sequence_len=d["sequence_len"], vocab_size=d["vocab_size"], n_embd=d["n_embd"],
            pad_vocab_size_to=d.get("pad_vocab_size_to", 64), reference=d.get("reference"),
            shared={k: ComponentSpec.from_dict(v) for k, v in d.get("shared", {}).items()},
            input=ComponentSpec.from_dict(d["input"]),
            body=ComponentSpec.from_dict(d["body"]),
            output=ComponentSpec.from_dict(d["output"]),
            adapters=[AdapterSpec.from_dict(a) for a in d.get("adapters", [])],
            frozen=list(d.get("frozen", [])),
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
