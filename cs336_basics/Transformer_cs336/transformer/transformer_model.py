import torch
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.nn import functional as F, init
from einops import rearrange, repeat, einsum, reduce
import math
from jaxtyping import Float, Int
from cs336_basics.Transformer_cs336.modules import Embedding
from cs336_basics.Transformer_cs336.modules import Linear
from cs336_basics.Transformer_cs336.modules import RotaryPositionEmbedding
from cs336_basics.Transformer_cs336.modules import RMSNorm, SwiGLU
from cs336_basics.Transformer_cs336.attention import MultiheadSelfAttention
from .transformer_block import TransformerBlock

class Transformer(torch.nn.Module):
    def __init__(self, d_model:int, num_heads:int, d_ff:int, vocab_size:int, context_length: int, num_layers:int,
                 max_seq_len:int = None, theta: float = None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.theta = theta

        self.embedding = Embedding(vocab_size, d_model)

        self.transformer_list = torch.nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff, max_seq_len, theta)
                                                    for _ in range(num_layers)])
        
        self.final_norm = RMSNorm(d_model)

        self.output_proj = Linear(d_model, vocab_size)

    def forward(self, x: Float[Tensor, "b s"]) -> Float[Tensor, "b s v"]:
        b, s = x.shape
        print ("xx shape:", x.shape)
        x = self.embedding(x)
        pos_ids = torch.arange(0, s)
        pos_ids = repeat(pos_ids, "s->b s", b = b)
        print ("pos_ids:", pos_ids.shape)

        for layer in self.transformer_list:
            x = layer(x, pos_ids)

        x = self.final_norm(x)
        x = self.output_proj(x)

        return x

        
