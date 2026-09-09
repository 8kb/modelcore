"""
Runtime: the small set of ambient values a modelcore component may need that aren't part of the
model's config -- what dtype to compute in, and where to send log lines. Everything in
modelcore/ is injected explicitly (a component asks the catalog for it via needs=("runtime",)),
never read off a module-level global -- see catalog.py's build context.

The host application's own COMPUTE_DTYPE-shaped global, if it has one, should source its value
from DEFAULT_RUNTIME below, not the other way around: modelcore has zero dependencies on its host,
so everything outside it adapts to modelcore's values instead of the reverse (see nanochat/common.py
for this repo's example).
"""
import os

import torch

_DTYPE_MAP = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
_ENV_VAR = "MODELCORE_DTYPE"
_LEGACY_ENV_VAR = "NANOCHAT_DTYPE"  # back-compat alias for this repo's original env var name


def detect_compute_dtype():
    """MODELCORE_DTYPE env override (NANOCHAT_DTYPE accepted as a back-compat alias), else CUDA
    capability, else fp32 (CPU/MPS)."""
    env = os.environ.get(_ENV_VAR) or os.environ.get(_LEGACY_ENV_VAR)
    if env is not None:
        return _DTYPE_MAP[env], f"set via {_ENV_VAR}={env}"
    if torch.cuda.is_available():
        # bf16 requires SM 80+ (Ampere: A100, A10, etc.)
        # Older GPUs like V100 (SM 70) and T4 (SM 75) only have fp16 tensor cores
        capability = torch.cuda.get_device_capability()
        if capability >= (8, 0):
            return torch.bfloat16, f"auto-detected: CUDA SM {capability[0]}{capability[1]} (bf16 supported)"
        # fp16 training requires GradScaler (not yet implemented), so fall back to fp32.
        # Users can still force fp16 via MODELCORE_DTYPE=float16 if they know what they're doing.
        return torch.float32, f"auto-detected: CUDA SM {capability[0]}{capability[1]} (pre-Ampere, bf16 not supported, using fp32)"
    # Note: MPS on recent macOS also handles bf16 fine, opt in via MODELCORE_DTYPE=bfloat16
    return torch.float32, "auto-detected: no CUDA (CPU/MPS)"


class Runtime:
    """The ambient values a component can ask the catalog to inject via needs=("runtime",)."""

    def __init__(self, compute_dtype=None, log=None):
        if compute_dtype is None:
            compute_dtype, reason = detect_compute_dtype()
        else:
            reason = "explicit"
        self.compute_dtype = compute_dtype
        self.compute_dtype_reason = reason
        self.log = log or (lambda msg: None)


DEFAULT_RUNTIME = Runtime()
