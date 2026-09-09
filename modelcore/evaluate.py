"""
evaluate_bpb: bits-per-byte evaluation for a base model. Internal module -- ModelManager.evaluate_bpb
is the public seam (mirrors new_decoder/generate.py's split: a free function here, wrapped by a
manager method).
"""
import math

import torch
import torch.distributed as dist

from modelcore.kernels.flash_attn import build_doc_args


@torch.no_grad()
def evaluate_bpb(model, batches, steps, token_bytes, bos_token_id=None, doc_masking_max_docs_per_row=None, padding_id=None):
    """
    Instead of the naive 'mean loss', this function returns the bits per byte (bpb),
    which is a tokenization vocab size-independent metric, meaning you are still comparing
    apples:apples if you change the vocab size. The way this works is that instead of just
    calculating the average loss as usual, you calculate the sum loss, and independently
    also the sum bytes (of all the target tokens), and divide. This normalizes the loss by
    the number of bytes that the target tokens represent.

    The added complexity is so that:
    1) All "normal" tokens are normalized by the length of the token in bytes
    2) No special tokens (e.g. <|bos|>) are included in the metric - they are masked out.
    3) No actively masked tokens (using ignore_index of e.g. -1) are included in the metric.

    token_bytes: a plain vector (list, numpy array, or tensor) of length vocab_size, giving the
    number of bytes for each token id, or 0 if the token is to not be counted (e.g. special
    tokens) -- a host application supplies this, e.g. from a datacore-prepared dataset's own
    token_bytes() (a caller reading its dataset via datacore.reader.batches's own tuple shape
    below already has the dataset handy to ask). Converted to a tensor on model's device once,
    here, so modelcore never has to depend on how the caller happened to obtain it.

    bos_token_id: when given, restricts attention to within each packed row's own document (see
    modelcore.kernels.flash_attn.build_doc_args) -- matches whatever masking training used, so
    val bpb stays comparable to the training loss it's evaluating. None (default) is unmasked, as
    before. doc_masking_max_docs_per_row: forwarded to build_doc_args's max_docs (as a per-row
    count, scaled by x's actual batch size) -- see build_doc_args's docstring for why its default
    is dataset-dependent, not a safe worst case. padding_id: the dataset's resolved pad-fill id
    (e.g. datacore.reader.DatasetInfo.padding_id), forwarded to build_doc_args so a pad-packed
    dataset's document boundaries are derived the same way training's own doc_args are, rather
    than always falling back to build_doc_args's bos-run heuristic.
    """
    device = model.get_device()
    token_bytes = torch.as_tensor(token_bytes, dtype=torch.int32, device=device)
    # record the losses
    total_nats = torch.tensor(0.0, dtype=torch.float32, device=device)
    total_bytes = torch.tensor(0, dtype=torch.int64, device=device)
    batch_iter = iter(batches)
    for _ in range(steps):
        x, y, *_ = next(batch_iter)  # a datacore-style loader yields a 3rd element (resume state)
        doc_args = None
        if bos_token_id is not None:
            max_docs = doc_masking_max_docs_per_row * x.size(0) if doc_masking_max_docs_per_row is not None else None
            doc_args = build_doc_args(x, bos_token_id, padding_id=padding_id, max_docs=max_docs)
        loss2d = model(x, y, loss_reduction='none', doc_args=doc_args) # (B, T)
        loss2d = loss2d.view(-1) # flatten
        y = y.view(-1) # flatten
        if (y.int() < 0).any(): # mps does not currently have kernel for < 0 for int64, only int32
            # slightly more complex code path if some target tokens are ignore_index (e.g. -1)
            # any target token < 0 is to be ignored: do NOT index token_bytes with negatives
            valid = y >= 0
            y_safe = torch.where(valid, y, torch.zeros_like(y))
            # map valid targets to their byte length; ignored targets contribute 0 bytes
            num_bytes2d = torch.where(
                valid,
                token_bytes[y_safe],
                torch.zeros_like(y, dtype=token_bytes.dtype)
            )
            total_nats += (loss2d * (num_bytes2d > 0)).sum()
            total_bytes += num_bytes2d.sum()
        else:
            # fast path: no ignored targets, safe to index directly
            num_bytes2d = token_bytes[y]
            total_nats += (loss2d * (num_bytes2d > 0)).sum()
            total_bytes += num_bytes2d.sum()
    # sum reduce across all ranks
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if world_size > 1:
        dist.all_reduce(total_nats, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_bytes, op=dist.ReduceOp.SUM)
    # move both to cpu, calculate bpb and return
    total_nats = total_nats.item()
    total_bytes = total_bytes.item()
    if total_bytes == 0:
        return float('inf')
    bpb = total_nats / (math.log(2) * total_bytes)
    return bpb
