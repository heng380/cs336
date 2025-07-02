import torch
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.nn import functional as F, init
from einops import rearrange, repeat, einsum, reduce
import math
from jaxtyping import Float, Int


def softmax(x:Tensor, i:int):
    x_max = torch.max(x, dim=i, keepdim=True).values
    x_exp = torch.exp(x-x_max)
    x_sum = torch.sum(x_exp, dim=i, keepdim=True)

    return x_exp/x_sum

def scaled_dot_product_attention(
        Q:Float[Tensor, "... seq_len d_k"],
        K:Float[Tensor, "... seq_len d_k"],
        V:Float[Tensor, "... seq_len d_v"],
        mask: Float[Tensor, "... seq_len seq_len"] | None=None
) -> Float[Tensor, "... seq_len d_v"]:
    QK = einsum(Q, K, "... seq_len1 d_k, ... seq_len2 d_k-> ... seq_len1 seq_len2")
    d_k = Q.shape[-1]
    QK = QK/math.sqrt(d_k)
    if mask is not None:
        QK = QK.masked_fill(mask==0, -torch.inf)
    attention = softmax(QK, -1)
    result = einsum(attention, V, "... seq_len1 seq_len2, ... seq_len2 d_v -> ... seq_len1 d_v")
    return result
