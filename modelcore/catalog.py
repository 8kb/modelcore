"""
Component registry: maps a type name -- the "#type" field of a modelcore.config.spec.ComponentSpec
-- to the nn.Module class that implements it, which build-context values it needs injected as
constructor kwargs, and (optionally) a semantic validator for its params. One flat namespace (no
"kind" of embedding/block/composer/shared) -- the parent deciding what a slot means is enough, and
type names are globally unique by convention.

A component self-registers via this decorator at its own class definition (see
modelcore/components/*.py, modelcore/composers/*.py) -- the catalog itself knows nothing about
any specific component, matching the module-ownership rule the rest of modelcore follows.
"""

_COMPONENT_REGISTRY = {}  # type name -> (cls, needs: tuple[str, ...], validate: callable | None)


def register_component(name, needs=(), validate=None):
    """Class decorator. `needs` names build-context values (see modelcore.model.Model's ctx:
    derived globals like n_embd/padded_vocab_size, plus built `shared` components like rope) this
    component's constructor requires, injected by name. `validate(params, ctx) -> list[str]` is an
    optional semantic check over this component's own params (e.g. "n_embd must be divisible by
    n_head") -- structural checks (unknown #type, unknown params, missing needs) are the catalog's
    own job, in modelcore.config.validate; a component's `validate` only needs to know about its
    own params, not the tree around it."""
    def decorator(cls):
        assert name not in _COMPONENT_REGISTRY, f"component type {name!r} already registered"
        _COMPONENT_REGISTRY[name] = (cls, tuple(needs), validate)
        return cls
    return decorator


def registered_types():
    return sorted(_COMPONENT_REGISTRY)


def get_component(name):
    """Returns (cls, needs, validate) for a registered type name, raising if unknown."""
    if name not in _COMPONENT_REGISTRY:
        raise ValueError(f"Unknown component type {name!r}. Registered: {registered_types()}")
    return _COMPONENT_REGISTRY[name]


def build_component(spec, ctx):
    """Build one ComponentSpec into a real nn.Module. Any ComponentSpec-valued param (or list of
    them) is resolved into a built module first -- so e.g. a composer receives its `blocks` param
    already built -- then the component class is constructed via cls(**resolved_params, **needed).
    Because a spec's params are already materialized to the component's exact constructor kwarg
    names, no per-component adapter layer is needed."""
    from modelcore.config.spec import ComponentSpec  # local import: spec.py has no need of us

    cls, needs, _validate = get_component(spec.type)

    kwargs = {}
    for key, value in spec.params.items():
        if isinstance(value, ComponentSpec):
            kwargs[key] = build_component(value, ctx)
        elif isinstance(value, list) and value and all(isinstance(v, ComponentSpec) for v in value):
            kwargs[key] = [build_component(v, ctx) for v in value]
        else:
            kwargs[key] = value
    for name in needs:
        assert name in ctx, f"component {spec.type!r} needs {name!r}, not present in the build context"
        kwargs[name] = ctx[name]
    return cls(**kwargs)
