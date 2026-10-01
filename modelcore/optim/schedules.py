"""
LR-multiplier / Muon-momentum step schedules: pure functions of the step index, no optimizer or
model in scope. The shapes are a good default, overridable via keyword args; the actual param-group mutation that reads these values is
ModelManager.apply_schedule, since it touches the optimizer's own on-disk format (see that
method's docstring).

momentum_warmup_steps is a parameter rather than a constant: a short run's warmup can otherwise eat
the whole horizon, so a caller may cap it (e.g. at num_iterations // 3).
"""


def lr_multiplier(it, num_iterations, warmup_steps, warmdown_ratio, final_lr_frac):
    """Linear warmup -> hold at 1.0 -> linear warmdown to final_lr_frac, over num_iterations
    steps. Multiply every param group's base ("initial") LR by this."""
    warmdown_iters = round(warmdown_ratio * num_iterations)
    if it < warmup_steps:
        return (it + 1) / max(warmup_steps, 1)
    elif it <= num_iterations - warmdown_iters:
        return 1.0
    else:
        progress = (num_iterations - it) / max(warmdown_iters, 1)
        return progress + (1 - progress) * final_lr_frac


def muon_momentum(it, num_iterations, warmdown_ratio, *, momentum_warmup_steps=400,
                   low=0.85, high=0.97, warmdown_low=0.90):
    """Warms up from `low` to `high` over the first momentum_warmup_steps steps, holds at `high`,
    then warms down to warmdown_low over the same warmdown window lr_multiplier uses. Sets Muon's
    own momentum directly (not a multiplier) -- see ModelManager.apply_schedule."""
    warmdown_iters = round(warmdown_ratio * num_iterations)
    warmdown_start = num_iterations - warmdown_iters
    momentum_warmup_steps = max(1, momentum_warmup_steps)
    if it < momentum_warmup_steps:
        frac = it / momentum_warmup_steps
        return (1 - frac) * low + frac * high
    elif it >= warmdown_start:
        progress = (it - warmdown_start) / max(warmdown_iters, 1)
        return high * (1 - progress) + warmdown_low * progress
    return high
