import torch
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.nn import functional as F, init
from einops import rearrange, repeat, einsum, reduce
import math
from .linear import Linear

class RMSNorm(torch.nn.Module):
    def __init__(self, d_model:int, eps:float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.reset_parameter()
    
    def reset_parameter(self):
        init.ones_(self.weight)

    def get_rms(self, x:Tensor) -> Tensor:
        MS = reduce(x ** 2, "b s d->b s 1", "mean")
        RMS = torch.sqrt(MS+self.eps)
        return RMS

    def forward(self, x:Tensor) -> Tensor:
        b, s, d = x.shape
        x = x.to(torch.float32)
        RMS = self.get_rms(x)
        x = x/RMS * self.weight
        return x
