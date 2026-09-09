"""
Test the parameter-role protocol (modelcore/roles.py) in isolation, with small hand-built
nn.Module trees -- no real model needed.

python -m pytest modelcore/tests/test_roles.py -v
"""

import pytest
import torch
import torch.nn as nn

from modelcore.components.linear import Linear
from modelcore.roles import build_param_groups, collect_param_roles


class _Declared(nn.Module):
    """A Linear (defaults to role "matrix") plus a bare Parameter declared via PARAM_ROLES."""
    PARAM_ROLES = {"scale": "smear"}

    def __init__(self):
        super().__init__()
        self.proj = Linear(4, 4, bias=False)
        self.scale = nn.Parameter(torch.zeros(1))


class _Undeclared(nn.Module):
    """A bare Parameter with no PARAM_ROLES entry -- should raise."""
    def __init__(self):
        super().__init__()
        self.proj = Linear(4, 4, bias=False)
        self.mystery = nn.Parameter(torch.zeros(1))


def test_declared_roles_are_collected():
    m = _Declared()
    roles = collect_param_roles(m)
    assert roles["matrix"] == [m.proj.weight]
    assert roles["smear"] == [m.scale]


def test_undeclared_non_linear_parameter_raises():
    m = _Undeclared()
    with pytest.raises(ValueError, match="mystery"):
        collect_param_roles(m)


class _Holder(nn.Module):
    """Wraps a Parameter so it can be planted as its own, separately-visited submodule -- this is
    what a tied weight looks like structurally (e.g. an LMHead holding the embedding's Parameter
    object): two distinct modules, each with the shared tensor in their own local _parameters
    dict, so torch's own within-a-single-call parameter dedup never gets a chance to hide the
    second reference from the walk."""
    def __init__(self, p):
        super().__init__()
        self.weight = p


def test_shared_parameter_same_role_is_fine():
    """A tied parameter reachable through two different submodules, both declaring the same role
    for it, does not raise and is not double-counted."""
    class Tied(nn.Module):
        PARAM_ROLES = {"a": "embedding", "b": "embedding"}

        def __init__(self, shared):
            super().__init__()
            self.a = _Holder(shared)
            self.b = _Holder(shared)

    shared = nn.Parameter(torch.zeros(4))
    m = Tied(shared)
    roles = collect_param_roles(m)
    assert roles["embedding"] == [shared]  # not duplicated


def test_shared_parameter_conflicting_role_raises():
    class Conflicting(nn.Module):
        PARAM_ROLES = {"a": "embedding", "b": "unembedding"}

        def __init__(self, shared):
            super().__init__()
            self.a = _Holder(shared)
            self.b = _Holder(shared)

    shared = nn.Parameter(torch.zeros(4))
    m = Conflicting(shared)
    with pytest.raises(ValueError, match="already claimed"):
        collect_param_roles(m)


def test_param_roles_override_bypasses_recursive_walk():
    """A module with a callable param_roles() method (the escape hatch) is trusted directly,
    even if its subtree would otherwise fail the recursive declaration check."""
    class Manual(nn.Module):
        def __init__(self):
            super().__init__()
            self.mystery = nn.Parameter(torch.zeros(1))  # would raise via the recursive path

        def param_roles(self):
            return {"smear": [self.mystery]}

    m = Manual()
    roles = collect_param_roles(m)
    assert roles["smear"] == [m.mystery]


def test_build_param_groups_splits_muon_role_by_shape_and_preserves_policy_order():
    p_a1 = nn.Parameter(torch.zeros(2, 3))
    p_a2 = nn.Parameter(torch.zeros(2, 3))
    p_b = nn.Parameter(torch.zeros(5, 1))
    p_scalar = nn.Parameter(torch.zeros(1))
    role_params = {
        "matrix": [p_a1, p_b, p_a2],  # two distinct shapes
        "smear": [p_scalar],
    }
    # policy order: smear before matrix, opposite of role_params insertion order -- output order
    # must follow policy, not role_params.
    policy = {
        "smear": dict(kind="adamw", lr=0.1),
        "matrix": dict(kind="muon", lr=0.02),
    }
    groups = build_param_groups(role_params, policy)
    assert len(groups) == 3  # 1 adamw group + 2 muon groups (one per shape)
    assert groups[0]["kind"] == "adamw" and groups[0]["params"] == [p_scalar]
    muon_groups = groups[1:]
    assert all(g["kind"] == "muon" for g in muon_groups)
    shapes_seen = [tuple(g["params"][0].shape) for g in muon_groups]
    assert shapes_seen == sorted(shapes_seen)
    assert {id(p) for g in muon_groups for p in g["params"]} == {id(p_a1), id(p_a2), id(p_b)}


def test_build_param_groups_raises_on_role_not_in_policy():
    role_params = {"mystery_role": [nn.Parameter(torch.zeros(1))]}
    with pytest.raises(ValueError, match="mystery_role"):
        build_param_groups(role_params, policy={})


def test_build_param_groups_skips_empty_roles():
    """A role present in the policy but with no parameters in this model produces no group."""
    groups = build_param_groups({}, policy={"matrix": dict(kind="muon", lr=0.02)})
    assert groups == []
