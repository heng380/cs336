import torch 
import torch.nn as nn
from torch.nn import functional as F, init
from einops import rearrange, einsum, reduce
from torch import Tensor

# class RotaryPositionEmbedding(nn.Module):

#     def __init__(
#         self,
#         theta: float,
#         d_k: int,    
#         max_seq_len:int,
#         device: torch.device | None = None
#     ) -> None:
#         super().__init__()
#         self.theta = theta
#         self.d_k = d_k
#         self.max_seq_len = max_seq_len
#         self.device = device

#         # Create the positional encodings
#         self.create_positional_encodings()

#     def create_positional_encodings(self) -> None:
       
#         # Compute inverse frequencies for each pair dimension
#         inv_freq = 1.0 / (
#             self.theta ** (torch.arange(0, self.d_k, 2, dtype=torch.float32, device=self.device) / self.d_k)
#         )  # (d_k/2,))

#         # Positions index
#         positions = torch.arange(self.max_seq_len, dtype=torch.float32, device=self.device)  # (max_seq_len,)

#         # Compute rotation angles: outer product -> (max_seq_len, d_k/2)
#         angles = positions[:, None] * inv_freq[None, :]
#         cos = angles.cos()  # (max_seq_len, d_k/2)
#         sin = angles.sin()  # (max_seq_len, d_k/2)

#         # Build the 2×2 rotation matrices for each position and head pair:
#         # rot_mats shape = (max_seq_len, d_k/2, 2, 2)
#         R = torch.stack([
#             torch.stack([cos, -sin], dim=-1),  # row 0
#             torch.stack([sin,  cos], dim=-1),  # row 1
#         ], dim=-2)

#         self.register_buffer("R", R, persistent=False)
#         return None


#     def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
#         x_even = x[..., 0::2]
#         x_odd  = x[..., 1::2]

#         # Gather the corresponding rotation matrices: (..., seq_len, d_k/2, 2, 2)
#         rot = self.R[token_positions]

#         # Pack x into last dimension: (..., seq_len, d_k/2, 2)
#         x_pair = torch.stack([x_even, x_odd], dim=-1)

#         # Matrix multiply each 2×2 rot matrix with its x_pair vector:
#         # (..., seq_len, d_k/2, 2, 1)
#         out_pair = torch.matmul(rot, x_pair.unsqueeze(-1))

#         # Remove trailing singleton: (..., seq_len, d_k/2, 2)
#         out_pair = out_pair.squeeze(-1)

#         # Flatten the last two dims back to d_k: (..., seq_len, d_k)
#         return out_pair.flatten(-2)
    
class RotaryPositionEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device
        self.create_positional_encodings()
    
    def create_positional_encodings(self):
        inv_freq = 1 / (self.theta ** (torch.arange(0, self.d_k, 2, device=self.device) / self.d_k))  # d/2
        positions = torch.arange(0, self.max_seq_len, device=self.device)  # max_seq_len
        angles = positions[:, None] * inv_freq[None, :]  # max_seq_len * d/2
        cos = angles.cos()
        sin = angles.sin()
        R = torch.stack([
            torch.stack([cos, -sin], dim=-1),
            torch.stack([sin, cos], dim=-1)
        ], dim=-2)

        self.register_buffer("R", R, persistent=False)  # max_seq_len, d/2, 2, 2

    def forward(self, x: Tensor, token_positions: Tensor):
        x_split = rearrange(x, "b h s (d_kk r2) -> b h s d_kk r2", r2=2)
        #x_split = rearrange(x, "... (d k) -> ... d k", k=2)
        rot = self.R[token_positions]
        print("R shape:", self.R.shape)
        print ("token_positions shape:", token_positions.shape)
        print ("x_split:", x_split.shape)
        print ("rot shape:", rot.shape)
        out_pair = einsum(
            rot,
            x_split,
              "b seq d_kk r1 r2, b h seq d_kk r2 -> b h seq d_kk r1"     # 可以显式加上 h 这个维度, 使用 rearrange 和 repeat
        )
        result = rearrange(out_pair, "b h seq d_kk r1 -> b h seq (d_kk r1)")
        return result