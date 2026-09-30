"""
Import every composer module so its @register_component decorator runs and populates
modelcore.catalog.
"""
from modelcore.composers import stack  # noqa: F401
