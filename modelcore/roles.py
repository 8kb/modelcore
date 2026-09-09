"""
Parameter-role protocol.

Every component must be able to say what *kind* of parameter each of its tensors is (a role
string like "matrix", "embedding", "smear", ...) so that optimizer grouping and parameter
accounting don't need a hand-maintained partition of self.parameters() that silently drifts as
submodules are added, moved, or reused. In particular: modelcore.optim.MuonAdamW routes 2D matrix
params through Muon and everything else through AdamW, and a parameter that lands in the wrong
bucket (e.g. a value-embedding table swept into Muon because it happened to live inside the same
submodule as the attention matrices) fails silently -- same parameter count, worse model.

A module declares roles for the parameters it *directly* owns via a PARAM_ROLES class attribute,
mapping an attribute name to a role string. The attribute can be a Parameter (that parameter takes
the role) or a child submodule (every parameter in that submodule's whole subtree takes the role --
the walk does not recurse further into it). collect_param_roles() walks the tree resolving these
declarations; a modelcore.components.linear.Linear's .weight defaults to role "matrix" when not
otherwise declared (matches the "Linear marks a matmul" convention modelcore.stats relies on);
anything else undeclared raises rather than falling through into a default bucket.

Escape hatch: a module can define param_roles(self) -> dict[str, list[Parameter]] instead of (or
as well as) PARAM_ROLES, for a module that wants to own the whole assignment for its subtree in one
place rather than one role per attribute.
"""

from collections import defaultdict

import torch.nn as nn

from modelcore.components.linear import Linear


def collect_param_roles(module: nn.Module) -> dict[str, list[nn.Parameter]]:
    """Return {role: [params]}, covering every parameter in module's subtree exactly once.
    Raises if a parameter's role can't be determined, or if two different roles both claim the
    same parameter (e.g. a weight-tied Parameter reachable through two different paths -- claiming
    it twice for the *same* role is fine and is how tying is expected to be expressed)."""
    roles: dict[str, list[nn.Parameter]] = defaultdict(list)
    assigned: dict[int, str] = {}  # id(param) -> role already claimed for it

    def claim(role, param):
        pid = id(param)
        prior = assigned.get(pid)
        if prior is not None:
            if prior != role:
                raise ValueError(f"parameter already claimed for role {prior!r}, cannot also claim {role!r}")
            return  # same role claimed twice (e.g. a tied parameter reached via two paths): fine
        assigned[pid] = role
        roles[role].append(param)

    def visit(m: nn.Module):
        override = getattr(m, "param_roles", None)
        if callable(override):
            for role, params in override().items():
                for p in params:
                    claim(role, p)
            return
        declared = getattr(type(m), "PARAM_ROLES", {})
        for name, child in list(m.named_children()):
            if name in declared:
                for p in child.parameters():
                    claim(declared[name], p)
            else:
                visit(child)
        for name, p in m.named_parameters(recurse=False):
            if name in declared:
                claim(declared[name], p)
            elif isinstance(m, Linear) and name == "weight":
                claim("matrix", p)
            else:
                raise ValueError(
                    f"parameter {type(m).__name__}.{name} has no declared role -- "
                    f"add it to {type(m).__name__}.PARAM_ROLES, or a param_roles() override"
                )

    visit(module)

    covered_ids = set(assigned)
    all_ids = {id(p) for p in module.parameters()}
    missing = all_ids - covered_ids
    assert not missing, f"{len(missing)} parameter(s) were not covered by any role"
    return dict(roles)


def build_param_groups(role_params: dict[str, list[nn.Parameter]], policy: dict[str, dict]) -> list[dict]:
    """Turn {role: [params]} (from collect_param_roles) plus an *ordered* {role: hyperparameters}
    policy into modelcore.optim.MuonAdamW param_groups.

    The order policy is iterated in becomes the on-disk param_group layout: optimizer state is
    checkpointed per-rank and reloaded positionally, so reordering this is a compatibility break,
    not just a style choice.

    A role with kind="muon" is split into one group per parameter shape -- Muon stacks same-shape
    params for its fused Newton-Schulz/Polar-Express step (see modelcore.optim.MuonAdamW)."""
    unknown = set(role_params) - set(policy)
    if unknown:
        raise ValueError(f"parameter roles present but not in the optimizer policy: {sorted(unknown)}")

    groups = []
    for role, hparams in policy.items():
        params = role_params.get(role, [])
        if not params:
            continue
        if hparams["kind"] == "muon":
            for shape in sorted({p.shape for p in params}):
                groups.append({**hparams, "params": [p for p in params if p.shape == shape]})
        else:
            groups.append({**hparams, "params": params})
    return groups
