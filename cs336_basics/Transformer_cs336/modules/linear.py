import torch
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.nn import functional as F, init
from einops import rearrange, repeat, einsum, reduce
import math


class Linear(torch.nn.Module):
    def __init__(self, in_features:int, out_features:int, device: torch.device=None, dtype: torch.dtype=None) -> None:
        super().__init__()

        factory_kwargs = {"device":device, "dtype":dtype}

        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(torch.ones(out_features, in_features, device=device, dtype=dtype))

        self.reset_parameter()

    def reset_parameter(self) -> None:
        std = math.sqrt(2/(self.in_features + self.out_features))
        init.trunc_normal_(self.weight, mean=0, std= std, a = -3*std, b = 3*std)

    def forward(self, x: Tensor) -> Tensor:
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")



def twosum(a, b):
    return a+b