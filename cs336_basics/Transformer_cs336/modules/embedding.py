import torch
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.nn import functional as F, init
from einops import rearrange, repeat, einsum, reduce
import math


class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings:int, embedding_dim:int, device:torch.device=None, dtype:torch.dtype=None):
        super().__init__()
        self.num_embedings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = Parameter(torch.ones(num_embeddings, embedding_dim, device=device, dtype=dtype))
        self.reset_parameter()
    
    def reset_parameter(self):
        init.trunc_normal_(self.weight, mean=0, std=1, a = -3, b = 3)

    def forward(self, token_ids:Tensor):
        return self.weight[token_ids]
