"""
State-space-duality (SSD) scan for Mamba-2-style mixers, with automatic kernel/reference switching.

    y, final_state = ssd_scan(x, dt, A, B, C, D, chunk_size, initial_state=None, doc_ids=None)
    y, new_state   = ssd_step(state, x, dt, A, B, C, D)          # one decode token

The model, per head (P = head_dim, N = d_state):  h_t = exp(dt_t * A) h_{t-1} + dt_t * x_t (x) B_t,
y_t = h_t . C_t + D * x_t, with h of shape (P, N). Shapes: x (B, T, H, P); dt (B, T, H), already
softplus'd and positive; A (H,), negative; B, C (B, T, G, N) with G groups dividing H; D (H,) or
None; state (B, H, P, N). doc_ids (B, T), optional: the recurrent state resets at every change of
document id, so a packed row equals its documents run alone (intra-document masking).

Two implementations:
- the pure-PyTorch chunked SSD below -- the reference, and what runs on CPU/MPS and whenever the
  kernel is unavailable. It resets state at document boundaries *exactly* (masks, not a -inf decay,
  whose cumulative sums would cancel catastrophically), so it is also what the kernel path is
  checked against.
- on CUDA, mamba_ssm's Triton chunk-scan and state-update kernels, loaded through the `kernels` hub
  (kernels-community/mamba-ssm) the same way flash_attn.py loads FA3. Untested on a CUDA-less
  machine: tests/test_ssm.py::TestKernelVsReference compares them on a GPU.
"""
import torch


def _load_ssm_kernel():
    """(module_or_None, reason). Mirrors flash_attn._load_flash_attention_3: the reason is kept so a
    fallback can say why."""
    if not torch.cuda.is_available():
        return None, "no CUDA device available"
    try:
        import os
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        from kernels import get_kernel, has_kernel
        repo = "kernels-community/mamba-ssm"
        if not has_kernel(repo):
            return None, f"kernels.has_kernel('{repo}') returned False for this GPU/torch/CUDA build"
        return get_kernel(repo), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


_kernel, SSM_KERNEL_LOAD_ERROR = _load_ssm_kernel()
HAS_SSM_KERNEL = _kernel is not None

# Test override: 'kernel', 'reference', or None (auto).
_override_impl = None


def _use_kernel(x):
    if _override_impl == "kernel":
        assert HAS_SSM_KERNEL, "Cannot override to the SSM kernel: not available on this hardware"
        return True
    if _override_impl == "reference":
        return False
    return HAS_SSM_KERNEL and x.is_cuda and x.dtype in (torch.bfloat16, torch.float16)


def _segsum(a):
    """(..., T) -> (..., T, T): out[i, j] = sum_{k=j+1..i} a_k for i >= j, -inf above the diagonal."""
    T = a.size(-1)
    a = a.unsqueeze(-1).expand(*a.shape, T)
    lower = torch.tril(torch.ones(T, T, dtype=torch.bool, device=a.device), diagonal=-1)
    seg = torch.cumsum(a.masked_fill(~lower, 0), dim=-2)
    return seg.masked_fill(~torch.tril(torch.ones(T, T, dtype=torch.bool, device=a.device)), -torch.inf)


def _ssd_core(X, a, B, C, chunk_size, initial_state=None, doc_ids=None):
    """The chunked scan on already-scaled inputs: h_t = exp(a_t) h_{t-1} + X_t (x) B_t, y_t = h_t . C_t
    (no D skip). X (B, T, H, P) is the input including its step-size factor, a (B, T, H) the log
    decay (<= 0), B/C (B, T, G, N). Returns (y float32, final_state float32)."""
    Bsz, T, H, P = X.shape
    G, N = B.size(2), B.size(3)
    X, a, B, C = (t.float() for t in (X, a, B, C))
    Q = chunk_size
    pad = (-T) % Q
    if pad:  # a = 0 and X = 0: decay 1, no input -- the tail is neutral for outputs and state
        X, a, B, C = (torch.nn.functional.pad(t, (0, 0) * (t.dim() - 2) + (0, pad)) for t in (X, a, B, C))
        if doc_ids is not None:  # the neutral tail stays in the last document
            doc_ids = torch.cat([doc_ids, doc_ids[:, -1:].expand(Bsz, pad)], dim=1)
    c = X.size(1) // Q
    B = B.repeat_interleave(H // G, dim=2)
    C = C.repeat_interleave(H // G, dim=2)

    X = X.view(Bsz, c, Q, H, P)
    Bc, Cc = B.view(Bsz, c, Q, H, N), C.view(Bsz, c, Q, H, N)
    a = a.view(Bsz, c, Q, H).permute(0, 3, 1, 2)          # (B, H, c, Q)
    a_cs = torch.cumsum(a, dim=-1)
    ids = None if doc_ids is None else doc_ids.view(Bsz, c, Q)

    # 1. within each chunk (the "diagonal" blocks)
    L = torch.exp(_segsum(a))                             # (B, H, c, Q, Q)
    if ids is not None:
        L = L * (ids.unsqueeze(-1) == ids.unsqueeze(-2)).unsqueeze(1)
    Y_diag = torch.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", Cc, Bc, L, X)

    # 2. each chunk's own final state (only tokens of the chunk's last document reach it)
    decay_states = torch.exp(a_cs[..., -1:] - a_cs)       # (B, H, c, Q)
    if ids is not None:
        decay_states = decay_states * (ids == ids[..., -1:]).unsqueeze(1)
    states = torch.einsum("bclhn,bhcl,bclhp->bchpn", Bc, decay_states, X)

    # 3. carry states across chunks; a document boundary inside (or at the start of) a chunk cuts
    # the carry from everything before it
    if initial_state is None:
        initial_state = torch.zeros_like(states[:, :1])
    else:
        initial_state = initial_state.float().unsqueeze(1)
    states = torch.cat([initial_state, states], dim=1)    # (B, c+1, H, P, N)
    decay_chunk = torch.exp(_segsum(torch.nn.functional.pad(a_cs[..., -1], (1, 0))))  # (B, H, c+1, c+1)
    if ids is not None:
        prev_last = torch.cat([ids[:, :1, 0], ids[:, :-1, -1]], dim=1)                # (B, c)
        reset = (ids[:, :, -1] != prev_last).long()
        epoch = torch.cumsum(torch.nn.functional.pad(reset, (1, 0)), dim=1)             # (B, c+1)
        decay_chunk = decay_chunk * (epoch.unsqueeze(-1) == epoch.unsqueeze(-2)).unsqueeze(1)
    new_states = torch.einsum("bhzk,bkhpn->bzhpn", decay_chunk, states)
    states_in, final_state = new_states[:, :-1], new_states[:, -1]

    # 4. what the state entering each chunk contributes to that chunk's outputs
    state_decay = torch.exp(a_cs)                          # (B, H, c, Q)
    if ids is not None:
        state_decay = state_decay * (ids == prev_last.unsqueeze(-1)).unsqueeze(1)
    Y_off = torch.einsum("bclhn,bchpn,bhcl->bclhp", Cc, states_in, state_decay)

    return (Y_diag + Y_off).reshape(Bsz, c * Q, H, P)[:, :T], final_state


def ssd_scan_reference(x, dt, A, B, C, D, chunk_size, initial_state=None, doc_ids=None):
    y, final_state = _ssd_core(x.float() * dt.float().unsqueeze(-1), dt.float() * A.float(), B, C,
                               chunk_size, initial_state, doc_ids)
    if D is not None:
        y = y + D.float().view(1, 1, -1, 1) * x.float()
    return y.to(x.dtype), final_state


def ssd_scan_decay(X, a, B, C, chunk_size, initial_state=None, doc_ids=None):
    """The same scan with the decay given directly instead of as dt * A: h_t = exp(a_t) h_{t-1} +
    X_t (x) B_t, y_t = h_t . C_t, for a token-dependent log decay a (B, T, H), a < 0, and inputs X
    (B, T, H, P) that already carry their own scale. What Mamba-3 needs (a data-dependent A and an
    input weight that is not the step size). No D skip. Returns (y, final_state float32).

    Kernel path: mamba_chunk_scan_combined takes a per-head scalar A, so A = -1 and dt = -a make
    the decay exp(a), and x is divided by dt to undo the kernel's own x * dt."""
    if _use_kernel(X):
        dt = (-a).float()
        y, final = _kernel.mamba_chunk_scan_combined(
            (X.float() / dt.unsqueeze(-1)).to(X.dtype), dt, -torch.ones(X.size(2), device=X.device), B, C, chunk_size,
            initial_states=None if initial_state is None else initial_state.to(X.dtype),
            seq_idx=None if doc_ids is None else doc_ids.to(torch.int32), dt_softplus=False, return_final_states=True)
        return y, final.float()
    y, final = _ssd_core(X, a, B, C, chunk_size, initial_state, doc_ids)
    return y.to(X.dtype), final


def ssd_scan(x, dt, A, B, C, D, chunk_size, initial_state=None, doc_ids=None):
    """See the module docstring. Returns (y (B, T, H, P), final_state (B, H, P, N) in float32)."""
    if _use_kernel(x):
        y, final = _kernel.mamba_chunk_scan_combined(
            x, dt.float(), A.float(), B, C, chunk_size, D=None if D is None else D.float(),
            initial_states=None if initial_state is None else initial_state.to(x.dtype),
            seq_idx=None if doc_ids is None else doc_ids.to(torch.int32),
            dt_softplus=False, return_final_states=True)
        return y, final.float()
    return ssd_scan_reference(x, dt, A, B, C, D, chunk_size, initial_state, doc_ids)


def ssd_step_reference(state, x, dt, A, B, C, D):
    H, G = x.size(1), B.size(1)
    dtype = x.dtype
    x, dt, A, B, C = (t.float() for t in (x, dt, A, B, C))
    Bh, Ch = B.repeat_interleave(H // G, dim=1), C.repeat_interleave(H // G, dim=1)   # (B, H, N)
    decay = torch.exp(dt * A).unsqueeze(-1).unsqueeze(-1)                              # (B, H, 1, 1)
    state = state.float() * decay + (dt.unsqueeze(-1) * x).unsqueeze(-1) * Bh.unsqueeze(2)
    y = torch.einsum("bhpn,bhn->bhp", state, Ch)
    if D is not None:
        y = y + D.float().view(1, H, 1) * x
    return y.to(dtype), state


def ssd_step(state, x, dt, A, B, C, D):
    """One decode token. x (B, H, P); dt (B, H); B, C (B, G, N); state (B, H, P, N). Returns
    (y (B, H, P), new_state)."""
    if _use_kernel(x):
        Bsz, H, P = x.shape
        N = state.size(-1)
        state = state.to(x.dtype)  # the kernel updates in place, in the state's own dtype
        y = _kernel.selective_state_update(
            state, x, dt.unsqueeze(-1).expand(Bsz, H, P), A.float().view(H, 1, 1).expand(H, P, N), B, C,
            None if D is None else D.float().view(H, 1).expand(H, P), dt_softplus=False)
        return y, state
    return ssd_step_reference(state, x, dt, A, B, C, D)
