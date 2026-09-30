"""
Test-only inverse of modelcore.config.upgrade.upgrade_v2_to_v3: turn a v3 config dict back into what
a v2 host would have written (gpt_block/plain_block/gated_*, the backout composer). It exists so the
upgrade can be checked as a round trip over every test flavor; it only needs to handle what those
flavors use (a gpt block never shares KV, features are the known ones), and asserts otherwise.
"""
import copy

import pytest

TYPE = "#type"


def _v2_can_express(cond):
    """v2 predates later mixers/features: a flavor using them has no v2 spelling, so a test that
    downgrades it is skipped (it has nothing to say about v2)."""
    if not cond:
        pytest.skip("flavor uses components newer than v2")


def _block_v3_to_v2(b):
    mixer = b["mixer"]
    _v2_can_express(mixer[TYPE] == "attention")
    block_feats = {f[TYPE]: f for f in b["features"]}
    mixer_feats = {f[TYPE]: f for f in mixer["features"]}
    is_gpt = "resid_lambdas" in block_feats
    _v2_can_express(set(block_feats) <= {"resid_lambdas"} and set(mixer_feats) <= {"value_embed", "output_gate"})
    out = {TYPE: ("gated_" if "output_gate" in mixer_feats else "") + ("gpt_block" if is_gpt else "plain_block"),
           **{k: v for k, v in b.items() if k.startswith("_")},
           "layer_idx": b["layer_idx"], **{k: mixer[k] for k in ("n_head", "n_kv_head", "head_dim", "window")}}
    if is_gpt:
        assert mixer["kv_slot"] is None and mixer["produces_kv"] is True
        assert mixer_feats.get("value_embed", {"gate_channels": 12})["gate_channels"] == 12
        out.update(has_value_embed="value_embed" in mixer_feats,
                   resid_lambda_init=block_feats["resid_lambdas"]["resid_lambda_init"],
                   x0_lambda_init=block_feats["resid_lambdas"]["x0_lambda_init"])
    else:
        assert "value_embed" not in mixer_feats
        out.update(kv_slot=mixer["kv_slot"], produces_kv=mixer["produces_kv"])
    out["mlp"] = b["ffn"]
    if "output_gate" in mixer_feats:
        out["attn_gate"] = {**mixer_feats["output_gate"], TYPE: "attn_gate"}
    return out


def _stack_v3_to_v2(s):
    feats = {f[TYPE]: f for f in s["features"]}
    assert set(feats) <= {"backout"}
    rest = {k: v for k, v in s.items() if k not in (TYPE, "features")}
    if "backout" in feats:
        return {TYPE: "backout", **rest,
                **{k: v for k, v in feats["backout"].items() if k != TYPE}}
    return {TYPE: "stack", **rest}


def _walk(node):
    if isinstance(node, dict):
        node = {k: _walk(v) for k, v in node.items()}
        if node.get(TYPE) == "block":
            return _block_v3_to_v2(node)
        if node.get(TYPE) == "stack":
            return _stack_v3_to_v2(node)
        return node
    if isinstance(node, list):
        return [_walk(v) for v in node]
    return node


def _fqn_v3_to_v2(name):
    return name.replace(".mixer.", ".attn.").replace(".ffn.", ".mlp.")


def downgrade_v3_to_v2(d):
    d = copy.deepcopy(d)
    d["format"] = "modelcore.v2"
    for section in ("shared", "input", "body", "output"):
        if section in d:
            d[section] = _walk(d[section])
    for a in d.get("adapters", []):
        a["target"] = _fqn_v3_to_v2(a["target"])
    if "frozen" in d:
        d["frozen"] = [_fqn_v3_to_v2(f) for f in d["frozen"]]
    return d
