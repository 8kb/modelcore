import torch.nn as nn
import torch.nn.functional as F


class Linear(nn.Linear):
    """nn.Linear that casts weights to match input dtype in forward.
    Replaces autocast: master weights stay fp32 for optimizer precision,
    but matmuls run in the activation dtype (typically bf16 from embeddings).

    Also the structural marker modelcore.stats.num_matmul_params uses to find every
    matmul-participating parameter in a model: any new matmul must go through this class
    (rather than a raw nn.Linear or nn.Parameter) or FLOPs accounting will silently miss it."""
    def forward(self, x):
        return F.linear(x, self.weight.to(dtype=x.dtype))
