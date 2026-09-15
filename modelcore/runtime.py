"""
Runtime: the small set of ambient values a modelcore component may need that aren't part of the
model's config -- what dtype to compute in, and where to send log lines. Everything in
modelcore/ is injected explicitly (a component asks the catalog for it via needs=("runtime",)),
never read off a module-level global -- see catalog.py's build context.

The host application's own COMPUTE_DTYPE-shaped global, if it has one, should source its value
from DEFAULT_RUNTIME below, not the other way around: modelcore has zero dependencies on its host,
so everything outside it adapts to modelcore's values instead of the reverse (see nanochat/common.py
for this repo's example).

Also home to compute_init/compute_cleanup (the device/seed/DDP bring-up every host's training/eval
entrypoint needs, moved here from two identical copies -- nanochat's nanochat/common.py and
tinylab's tinylab/runtime.py) and the GPU peak-FLOPs/peak-bandwidth tables (MFU/MBU denominators;
previously only in nanochat/common.py). Neither is part of the injected-runtime-value contract
above (no component ever needs a device or a peak-flops number); they live here because this is
where a host's own runtime-bringup module already imports COMPUTE_DTYPE from, not because they're
part of the Runtime/DEFAULT_RUNTIME class.
"""
import os

import torch
import torch.distributed as dist

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


# -----------------------------------------------------------------------------
# Device/DDP/seed bring-up. Ported from two identical copies (nanochat/nanochat/common.py,
# tinylab/tinylab/runtime.py) -- moved here rather than split into a new module because both
# copies already lived alongside their host's own COMPUTE_DTYPE re-export. compute_init's return
# shape (a 5-tuple) is kept exactly as both hosts already unpack it, so this is a pure import-path
# change at every existing call site.


def is_ddp_requested() -> bool:
    """True if launched by torchrun (env present), even before init."""
    return all(k in os.environ for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))


def is_ddp_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_dist_info():
    if is_ddp_requested():
        return True, int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])
    return False, 0, 0, 1


def autodetect_device_type(log=None):
    if torch.cuda.is_available():
        device_type = "cuda"
    elif torch.backends.mps.is_available():
        device_type = "mps"
    else:
        device_type = "cpu"
    (log or (lambda msg: None))(f"Autodetected device type: {device_type}")
    return device_type


def compute_init(device_type="cuda", *, seed=42, backend="nccl", log=None):
    """Basic device/seed/DDP initialization, shared by every op. device_type: cuda|cpu|mps, or
    "auto"/"" to autodetect. `log`, if given, receives the autodetection message (rank-0-only
    printing, if desired, is the caller's job -- see each host's own print0). Returns
    (is_ddp, ddp_rank, ddp_local_rank, ddp_world_size, device)."""
    if device_type in ("auto", ""):
        device_type = autodetect_device_type(log=log)
    assert device_type in ("cuda", "mps", "cpu"), f"Invalid device type: {device_type}"
    if device_type == "cuda":
        assert torch.cuda.is_available(), "device_type='cuda' but CUDA is not available"
    if device_type == "mps":
        assert torch.backends.mps.is_available(), "device_type='mps' but MPS is not available"

    torch.manual_seed(seed)
    if device_type == "cuda":
        torch.cuda.manual_seed(seed)
        torch.set_float32_matmul_precision("high")

    is_ddp_req, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    if is_ddp_req and device_type == "cuda":
        device = torch.device("cuda", ddp_local_rank)
        torch.cuda.set_device(device)
        dist.init_process_group(backend=backend, device_id=device)
        dist.barrier()
    else:
        device = torch.device(device_type)

    return is_ddp_req, ddp_rank, ddp_local_rank, ddp_world_size, device


def compute_cleanup():
    if is_ddp_initialized():
        dist.destroy_process_group()


# -----------------------------------------------------------------------------
# GPU peak-FLOPs/peak-bandwidth tables: MFU/MBU denominators. Ported from
# nanochat/nanochat/common.py (tinylab dropped these when it was ported -- see tinylab/AGENTS.md).
# Pure hardware facts, extendable/replaceable by a caller (pass a different table to the lookup
# functions, or just index the tuples directly).
# inspired by torchtitan: https://github.com/pytorch/torchtitan/blob/main/torchtitan/tools/utils.py
# and PR: https://github.com/karpathy/nanochat/pull/147

# hardcoded BF16 peak flops for various GPUs. Table order matters: more specific patterns first.
PEAK_FLOPS_TABLE = (
    # NVIDIA Blackwell
    (["gb200"], 2.5e15),
    (["grace blackwell"], 2.5e15),
    (["b200"], 2.25e15),
    (["b100"], 1.8e15),
    # NVIDIA Hopper
    (["h200", "nvl"], 836e12),
    (["h200", "pcie"], 836e12),
    (["h200"], 989e12),
    (["h100", "nvl"], 835e12),
    (["h100", "pcie"], 756e12),
    (["h100"], 989e12),
    (["h800", "nvl"], 989e12),
    (["h800"], 756e12),
    # NVIDIA Ampere data center
    (["a100"], 312e12),
    (["a800"], 312e12),
    (["a40"], 149.7e12),
    (["a30"], 165e12),
    # NVIDIA Ada data center
    (["l40s"], 362e12),
    (["l40-s"], 362e12),
    (["l40 s"], 362e12),
    (["l4"], 121e12),
    # AMD CDNA accelerators
    (["mi355"], 2.5e15),
    (["mi325"], 1.3074e15),
    (["mi300x"], 1.3074e15),
    (["mi300a"], 980.6e12),
    (["mi250x"], 383e12),
    (["mi250"], 362.1e12),
    # Consumer RTX
    (["5090"], 209.5e12),
    (["4090"], 165.2e12),
    (["3090"], 71e12),
)

# Peak HBM/GDDR memory bandwidth in bytes/sec. The decode phase of inference is
# memory-bandwidth-bound, so this is the roofline for tokens/sec (see MBU).
PEAK_BANDWIDTH_TABLE = (
    # NVIDIA Blackwell (HBM3e)
    (["gb200"], 8.0e12),
    (["grace blackwell"], 8.0e12),
    (["b200"], 8.0e12),
    (["b100"], 8.0e12),
    # NVIDIA Hopper
    (["h200"], 4.8e12),
    (["h100", "nvl"], 3.9e12),
    (["h100", "pcie"], 2.0e12),
    (["h100"], 3.35e12),  # SXM
    (["h800", "pcie"], 2.0e12),
    (["h800"], 3.35e12),  # SXM
    # NVIDIA Ampere data center (A100 80GB; the 40GB variant is 1.6e12)
    (["a100"], 2.0e12),
    (["a800"], 2.0e12),
    (["a40"], 696e9),
    (["a30"], 933e9),
    # NVIDIA Ada data center
    (["l40s"], 864e9),
    (["l40-s"], 864e9),
    (["l40 s"], 864e9),
    (["l4"], 300e9),
    # AMD CDNA accelerators
    (["mi355"], 8.0e12),
    (["mi325"], 6.0e12),
    (["mi300x"], 5.3e12),
    (["mi300a"], 5.3e12),
    (["mi250x"], 3.28e12),
    (["mi250"], 3.28e12),
    # Consumer RTX
    (["5090"], 1.79e12),
    (["4090"], 1.01e12),
    (["3090"], 936e9),
)


def peak_flops(device_name: str, table=PEAK_FLOPS_TABLE, log=None) -> float:
    """Unknown GPU -> inf, so MFU shows as 0% rather than a wrong guess."""
    name = device_name.lower()
    for patterns, flops in table:
        if all(p in name for p in patterns):
            return flops
    if "data center gpu max 1550" in name:
        # Ponte Vecchio (PVC) - dynamic based on compute units
        max_comp_units = torch.xpu.get_device_properties("xpu").max_compute_units
        return 512 * max_comp_units * 1300 * 10**6
    (log or (lambda msg: None))(f"Peak flops undefined for: {device_name}, MFU will show as 0%")
    return float("inf")


def peak_bandwidth(device_name: str, table=PEAK_BANDWIDTH_TABLE, log=None) -> float:
    """Unknown GPU -> inf, so MBU shows as 0% rather than a wrong guess."""
    name = device_name.lower()
    for patterns, bandwidth in table:
        if all(p in name for p in patterns):
            return bandwidth
    (log or (lambda msg: None))(f"Peak bandwidth undefined for: {device_name}, MBU will show as 0%")
    return float("inf")
