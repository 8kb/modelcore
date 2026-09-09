"""
Import every component module so its @register_component decorator runs and populates
modelcore.catalog. Each component self-registers at its own class definition -- this package
just needs to import them all once.
"""
from modelcore.components import attention, block, embedding, linear, mlp, norm, rope, rotary, unembedding  # noqa: F401
