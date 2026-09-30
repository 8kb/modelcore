"""
The one place in modelcore allowed to supply a default value: upgrade_v1_to_v2 turns a
"modelcore.v1" dict into a v2 one, and upgrade_v2_to_v3 (below) restructures a v2 dict into the
feature-based v3 shape without supplying any new architecture-affecting value. v1 left a lot implicit -- an MLP that was hardcoded per block
type, a norm that was a free function, constructor defaults like softcap=15 -- and v2 states every
one of those explicitly, because a config tree carries only concrete, already-decided values
(see modelcore/AGENTS.md). This module's whole job is to write today's implicit values out, so a
v1 config keeps building the exact same model.

Nothing else in modelcore may default an architecture-affecting value: a v3 dict missing one is
rejected (ModelConfig.from_dict / validate_config), not silently completed.

Pure dict surgery -- never builds a model. Walks the tree generically (every dict carrying
"#type", at any depth), so nested composers and any block arrangement work without this module
knowing about them.
"""
import copy
import re

from modelcore.config.spec import COMMENT_PREFIX, FORMAT, FORMAT_V1, FORMAT_V2, TYPE_KEY

_V1_TEMPLATE = "base"
_V1_PAD_VOCAB_SIZE_TO = 64
_GATED_MLP_MULTIPLE_OF = 256  # SwiGLUMLP's old multiple_of default, which block.py never overrode


def _gpt_mlp(n_embd: int) -> dict:
    """What gpt_block hardcoded: MLP(n_embd) -- 4x expansion, ReLU squared."""
    return {TYPE_KEY: "mlp", "activation": "relu2", "hidden_dim": 4 * n_embd}


def _plain_mlp(n_embd: int) -> dict:
    """What plain_block hardcoded: SwiGLUMLP(n_embd) -- 2/3 of a 4x expansion, rounded up."""
    hidden = int(2 * (4 * n_embd) / 3)
    hidden = _GATED_MLP_MULTIPLE_OF * ((hidden + _GATED_MLP_MULTIPLE_OF - 1) // _GATED_MLP_MULTIPLE_OF)
    return {TYPE_KEY: "gated_mlp", "activation": "silu", "hidden_dim": hidden}


def _upgrade_spec(spec: dict, n_embd: int, sequence_len: int) -> None:
    kind = spec[TYPE_KEY]
    if kind in ("gpt_block", "plain_block"):
        spec.setdefault("mlp", _gpt_mlp(n_embd) if kind == "gpt_block" else _plain_mlp(n_embd))
        # v1 had no head_dim concept at all -- every block derived it as n_embd // n_head. null is
        # the v2 spelling of "derive it" (see CausalSelfAttention), so this is a like-for-like
        # translation, not a new default value.
        spec.setdefault("head_dim", None)
        # Full attention was written window=sequence_len (the host preset layer's long window); it
        # is window=-1 now. Same result for any input up to sequence_len.
        window = spec.get("window")
        if isinstance(window, int) and window >= sequence_len:
            spec["window"] = -1
    if kind == "plain_block":
        spec.setdefault("kv_slot", None)
        spec.setdefault("produces_kv", True)
    elif kind == "lm_head":
        spec.setdefault("softcap", 15)
    elif kind == "token_embedding":
        spec.setdefault("smear", True)
    elif kind == "rotary":
        spec.setdefault("over_compute", 10)
    elif kind == "backout":
        spec.setdefault("backout_lambda_init", 0.2)


def _walk(node, n_embd: int, sequence_len: int) -> None:
    if isinstance(node, dict):
        if TYPE_KEY in node:
            _upgrade_spec(node, n_embd, sequence_len)
        for value in node.values():
            _walk(value, n_embd, sequence_len)
    elif isinstance(node, list):
        for value in node:
            _walk(value, n_embd, sequence_len)


def upgrade_v1_to_v2(d: dict) -> dict:
    d = copy.deepcopy(d)
    d["format"] = FORMAT_V2
    d.setdefault("template", _V1_TEMPLATE)
    d.setdefault("pad_vocab_size_to", _V1_PAD_VOCAB_SIZE_TO)
    # v1's norm was F.rms_norm(x, (C,)) with eps left unset -- which torch resolves to the input
    # dtype's machine epsilon. `"eps": None` is that, exactly (see components/norm.py).
    d.setdefault("shared", {}).setdefault("norm", {TYPE_KEY: "rms_norm", "eps": None})
    for section in ("shared", "input", "body", "output"):
        if section in d:
            _walk(d[section], d["n_embd"], d["sequence_len"])
    for adapter in d.get("adapters", []):
        adapter.setdefault("enabled", True)
    return d


# ---------------------------------------------------------------------------------------------------
# v2 -> v3: one Block class, a mixer slot, a features list. Pure restructuring -- every value is
# carried over, `null`/`true`/`[]` are only the spellings of what v2 blocks hardcoded.

V2_BLOCK_TYPES = ("gpt_block", "plain_block", "gated_gpt_block", "gated_plain_block")
# `backout` was a composer in v2 and is a stack feature in v3 -- the same name, told apart by the
# composer's `blocks` param.


def _comments(d: dict) -> dict:
    return {k: v for k, v in d.items() if k.startswith(COMMENT_PREFIX)}


def _upgrade_block_v2_to_v3(d: dict) -> None:
    kind = d[TYPE_KEY]
    comments = _comments(d)
    params = {k: v for k, v in d.items() if k != TYPE_KEY and k not in comments}

    def take(*names):
        return {n: params.pop(n) for n in names if n in params}

    is_gpt = kind.endswith("gpt_block")
    layer_idx = take("layer_idx")
    mixer = {TYPE_KEY: "attention", **take("n_head", "n_kv_head", "head_dim", "window")}
    if is_gpt:  # gpt_block had no KV-sharing params: always its own slot at its layer_idx
        mixer.update(kv_slot=None, produces_kv=True)
    else:
        mixer.update(take("kv_slot", "produces_kv"))
    mixer_features = []
    if params.pop("has_value_embed", False):
        mixer_features.append({TYPE_KEY: "value_embed", "gate_channels": 12})  # v2 hardcoded 12
    gate = params.pop("attn_gate", None)
    if isinstance(gate, dict):
        mixer_features.append({**gate, TYPE_KEY: "output_gate"})
    mixer["features"] = mixer_features
    block_features = []
    if is_gpt:
        block_features.append({TYPE_KEY: "resid_lambdas", **take("resid_lambda_init", "x0_lambda_init")})
    ffn = take("mlp")
    d.clear()
    d.update({TYPE_KEY: "block", **comments, **layer_idx, "mixer": mixer, **({"ffn": ffn["mlp"]} if ffn else {}),
              "features": block_features, **params})  # leftover (unknown) params stay, so validation names them


def _upgrade_stack_v2_to_v3(d: dict) -> None:
    kind = d[TYPE_KEY]
    comments = _comments(d)
    params = {k: v for k, v in d.items() if k != TYPE_KEY and k not in comments}
    features = []
    if kind == "backout":
        features.append({TYPE_KEY: "backout", **{k: params.pop(k) for k in ("backout_layer", "backout_lambda_init")
                                                if k in params}})
    d.clear()
    d.update({TYPE_KEY: "stack", **comments, **params, "features": features})


def _walk_v2_to_v3(node) -> None:
    if isinstance(node, dict):
        for value in list(node.values()):
            _walk_v2_to_v3(value)
        kind = node.get(TYPE_KEY)
        if kind in V2_BLOCK_TYPES:
            _upgrade_block_v2_to_v3(node)
        elif kind == "stack" or (kind == "backout" and "blocks" in node):
            _upgrade_stack_v2_to_v3(node)
    elif isinstance(node, list):
        for value in node:
            _walk_v2_to_v3(value)


_BLOCK_NAME = re.compile(r"^(?P<pre>(?:.*\.)?blocks\.\d+)\.(?P<rest>.*)$")


def _head_swap(rest: str, old: str, new: str):
    """`new + tail` if rest is `old` or starts with `old.`, else None."""
    if rest == old or rest.startswith(old + "."):
        return new + rest[len(old):]
    return None


def remap_v2_name(name: str) -> str:
    """A v2 module FQN or state-dict key -> its v3 spelling: attn -> mixer, mlp -> ffn, the gate,
    value embedding and lambdas moved under `features` (a ModuleDict keyed by feature `#type`).
    Names it has no rule for pass through unchanged, so it is safe on every key of a state dict and
    on a module FQN alike (adapter targets, `frozen`). Shared by upgrade_v2_to_v3 (config FQNs) and
    modelcore.convert (checkpoint keys) -- the rules must agree."""
    if name == "body.backout_lambda":
        return "body.features.backout.backout_lambda"
    m = _BLOCK_NAME.match(name)
    if m is None:
        return name
    pre, rest = m["pre"], m["rest"]
    for old, new in (("attn.gate", "mixer.features.output_gate"),
                     ("attn.value_embed", "mixer.features.value_embed.embed"),
                     ("attn.ve_gate", "mixer.features.value_embed.gate"),
                     ("attn", "mixer"), ("mlp", "ffn")):
        swapped = _head_swap(rest, old, new)
        if swapped is not None:
            return f"{pre}.{swapped}"
    if rest in ("resid_lambda", "x0_lambda"):
        return f"{pre}.features.resid_lambdas.{rest}"
    return name


def upgrade_v2_to_v3(d: dict) -> dict:
    d = copy.deepcopy(d)
    d["format"] = FORMAT
    for section in ("shared", "input", "body", "output"):
        if section in d:
            _walk_v2_to_v3(d[section])
    for adapter in d.get("adapters", []):
        if isinstance(adapter, dict) and "target" in adapter:
            adapter["target"] = remap_v2_name(adapter["target"])
    if "frozen" in d:
        d["frozen"] = [remap_v2_name(fqn) for fqn in d["frozen"]]
    return d


def upgrade_to_v3(d: dict) -> dict:
    """Any supported older dict -> v3 (a v3 dict comes back as a copy)."""
    fmt = d.get("format", FORMAT_V1)
    if fmt == FORMAT_V1:
        d = upgrade_v1_to_v2(d)
        fmt = FORMAT_V2
    if fmt == FORMAT_V2:
        d = upgrade_v2_to_v3(d)
    return copy.deepcopy(d)


def has_v2_types(config) -> bool:
    """True if a ModelConfig's tree still uses a v2-only component type (a host that builds v2
    ComponentSpecs directly). ModelConfig.to_dict stamps such a tree v2 so it upgrades on load."""
    from modelcore.config.spec import ComponentSpec

    def scan(value):
        if isinstance(value, ComponentSpec):
            return (value.type in V2_BLOCK_TYPES or (value.type == "backout" and "blocks" in value.params)
                    or any(scan(v) for v in value.params.values()))
        if isinstance(value, list):
            return any(scan(v) for v in value)
        return False

    return any(scan(s) for s in (config.input, config.body, config.output, *config.shared.values()))
