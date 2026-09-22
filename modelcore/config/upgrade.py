"""
The one place in modelcore allowed to supply a default value: upgrade_v1_to_v2 turns a
"modelcore.v1" dict into a v2 one. v1 left a lot implicit -- an MLP that was hardcoded per block
type, a norm that was a free function, constructor defaults like softcap=15 -- and v2 states every
one of those explicitly, because a config tree carries only concrete, already-decided values
(see modelcore/AGENTS.md). This module's whole job is to write today's implicit values out, so a
v1 config keeps building the exact same model.

Nothing else in modelcore may default an architecture-affecting value: a v2 dict missing one is
rejected (ModelConfig.from_dict / validate_config), not silently completed.

Pure dict surgery -- never builds a model. Walks the tree generically (every dict carrying
"#type", at any depth), so nested composers and any block arrangement work without this module
knowing about them.
"""
import copy

from modelcore.config.spec import FORMAT, TYPE_KEY

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
    d["format"] = FORMAT
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
