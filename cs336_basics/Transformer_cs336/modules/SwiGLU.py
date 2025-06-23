import torch
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.nn import functional as F, init
from einops import rearrange, repeat, einsum, reduce
import math
from .linear import Linear

class SwiGLU(torch.nn.Module):
    def __init__(self, d_model:int, d_ff:int, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff, d_model, device, dtype)
        self.w3 = Linear(d_model, d_ff, device, dtype)

    def forward(self, x:Tensor):
        x1 = self.w1(x)
        x1 = x1 * torch.sigmoid(x1)
        x3 = self.w3(x)
        x = einsum(x1, x3, "b s d, b s d->b s d")
        return self.w2(x)