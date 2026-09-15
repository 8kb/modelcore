"""
Scaling-law derivation: given parameter/FLOPs counts for a model (and its d12 muP reference),
derive the training horizon (iterations/tokens), batch size, and the learning-rate/weight-decay
corrections that follow from it. Pure math, no I/O, architecture-agnostic -- it only needs scalar
counts, not a model, so it computes the same numbers whether or not anything actually trains.

Moved here from two identical copies (nanochat's nanochat/scaling.py, tinylab's own
tinylab/ops/train.py) -- every input is a plain scalar and every output feeds
modelcore.OptimizerHparams, so this was training-loop policy duplicated across hosts rather than
anything host-specific. nanochat/nanochat/scaling.py is now a backward-compat re-export shim.

Ref: https://arxiv.org/abs/2505.13738 (Power Lines, optimal batch size scaling)
Ref: https://arxiv.org/abs/2405.13698 (T_epoch weight decay scaling)
"""

import math
from dataclasses import dataclass

B_REF = 2**19  # optimal batch size at d12 ~= 524,288 tokens (measured empirically)


@dataclass
class TrainingPlan:
    target_tokens: int          # optimal tokens for this model, per target_param_data_ratio
    total_batch_size: int       # tokens per optimizer step
    auto_batch_size: bool       # True if total_batch_size was auto-computed, not pinned by the caller
    batch_lr_scale: float       # multiply AdamW/Muon LRs by this for the chosen batch size
    weight_decay_scaled: float  # T_epoch-corrected weight decay for the chosen batch size/horizon
    num_iterations: int
    horizon_source: str         # "user" | "target_flops" | "target_param_data_ratio"
    total_tokens: int           # total_batch_size * num_iterations
    total_flops: float          # num_flops_per_token * total_tokens


def derive_training_plan(*, num_scaling_params, d_ref_scaling_params, num_flops_per_token,
                          target_param_data_ratio, target_flops, num_iterations,
                          total_batch_size, weight_decay, b_ref=B_REF):
    """All scalar inputs. num_iterations/total_batch_size are the caller's raw request (-1 means
    "not given"). b_ref overrides B_REF -- an empirically measured constant, not a derived law, so
    a caller re-measuring it for their own setup can pass a different value."""
    target_tokens = int(target_param_data_ratio * num_scaling_params)
    D_REF = target_param_data_ratio * d_ref_scaling_params  # compute-optimal d12 training horizon in tokens

    # Optimal batch size grows as ~D^0.383 (Power Lines): if D doubles from d12 to d24, B should
    # grow by 2^0.383 ~= 1.3x. Clamp to the nearest power of 2 for efficiency.
    auto_batch_size = total_batch_size == -1
    if auto_batch_size:
        batch_size_ratio = target_tokens / D_REF
        predicted_batch_size = b_ref * batch_size_ratio ** 0.383
        total_batch_size = 2 ** round(math.log2(predicted_batch_size))

    # AdamW: sqrt scaling is standard: eta ~ sqrt(B/B_ref). Muon uses the same scaling (not
    # studied carefully, assumption!). SGD's linear scaling doesn't apply -- not used here.
    batch_ratio = total_batch_size / b_ref
    batch_lr_scale = batch_ratio ** 0.5 if batch_ratio != 1.0 else 1.0

    # T_epoch framework (arxiv 2405.13698): T_epoch = B/(eta*lambda*D) held constant, combined
    # with the eta ~ sqrt(B/B_ref) scaling above, gives lambda = lambda_ref * sqrt(B/B_ref) * (D_ref/D).
    # Studied for AdamW, not Muon -- blindly following AdamW theory here too.
    weight_decay_scaled = weight_decay * math.sqrt(total_batch_size / b_ref) * (D_REF / target_tokens)

    # num_iterations: either given directly, or derived from target FLOPs, or from the target
    # data:param ratio (in that precedence order).
    if num_iterations > 0:
        horizon_source = "user"
    elif target_flops > 0:
        horizon_source = "target_flops"
        num_iterations = round(target_flops / (num_flops_per_token * total_batch_size))
    elif target_param_data_ratio > 0:
        horizon_source = "target_param_data_ratio"
        num_iterations = target_tokens // total_batch_size
    else:
        raise ValueError("No training horizon specified")

    total_tokens_actual = total_batch_size * num_iterations
    total_flops = num_flops_per_token * total_tokens_actual

    return TrainingPlan(
        target_tokens=target_tokens,
        total_batch_size=total_batch_size,
        auto_batch_size=auto_batch_size,
        batch_lr_scale=batch_lr_scale,
        weight_decay_scaled=weight_decay_scaled,
        num_iterations=num_iterations,
        horizon_source=horizon_source,
        total_tokens=total_tokens_actual,
        total_flops=total_flops,
    )
