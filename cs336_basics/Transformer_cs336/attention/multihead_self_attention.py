import torch
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.nn import functional as F, init
from einops import rearrange, repeat, einsum, reduce
import math
from jaxtyping import Float, Int
from cs336_basics.Transformer_cs336.modules import Linear
from cs336_basics.Transformer_cs336.modules import RotaryPositionEmbedding
from .attention_fn import *

class MultiheadSelfAttention(torch.nn.Module):
    def __init__(self, d_model:int, num_heads:int, max_seq_len:int = None, theta:float = None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q_proj_weight = Linear(d_model, self.num_heads*self.d_k)
        self.k_proj_weight = Linear(d_model, self.num_heads*self.d_k)
        self.v_proj_weight = Linear(d_model, self.num_heads*self.d_k)
        self.o_proj_weight = Linear(self.num_heads*self.d_k, d_model)

        self.max_seq_len = max_seq_len

        self.theta = theta

    def forward(self, x:Tensor, token_positions: Tensor = None):
        b, s, d_model = x.shape
        print ("shape of x:", x.shape)
        Q = self.q_proj_weight(x)
        K = self.k_proj_weight(x)
        V = self.v_proj_weight(x)
        print ("shape of q:", Q.shape)
        print ("head: ", self.num_heads)
        Q = rearrange(Q, "b s (h d_k)->b h s d_k", h=self.num_heads)
        K = rearrange(K, "b s (h d_k)->b h s d_k", h=self.num_heads)
        V = rearrange(V, "b s (h d_k)->b h s d_k", h=self.num_heads)
        

        self.mask = torch.tril(torch.ones(s,s)) # 下三角为 1

        if token_positions is not None and self.theta is not None and self.max_seq_len is not None:
            rope = RotaryPositionEmbedding(self.theta, self.d_k, self.max_seq_len)
            Q = rope(Q, token_positions)
            K = rope(K, token_positions)

        attn_output = scaled_dot_product_attention(Q, K, V, self.mask)

        attn_output = rearrange(attn_output, "b h s d->b s (h d)")

        return self.o_proj_weight(attn_output)
