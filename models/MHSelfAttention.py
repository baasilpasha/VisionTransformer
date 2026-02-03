# Implement self attention for the Vision Transformer Model
import torch
from einops import rearrange
from torch import nn


class MHSelfAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=None):
        super().__init__()
        self.heads = heads
        self.dim_head = dim // heads if dim_head is None else dim_head
        _dim = self.dim_head * heads
        self.to_qkv = nn.Linear(dim, _dim * 3, bias=False)
        self.W_out = nn.Linear(_dim, dim, bias=False)
        self.scale_factor = self.dim_head ** -0.5

    def forward(self, x, mask=None):
        # x shape: (batch_size, num_patches+1, dim)
        assert x.dim() == 3
        qkv = self.to_qkv(x)  # [b, n, 3 * heads * dim_head]
        qkv = rearrange(qkv, 'b n (three heads dim) -> three b heads n dim', three=3, heads=self.heads, dim=self.dim_head)
        q, k, v = qkv[0], qkv[1], qkv[2]

        scaled_dot_prod = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale_factor
        attention = torch.softmax(scaled_dot_prod, dim=-1)
        out = torch.einsum('b h i j, b h j d -> b h i d', attention, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.W_out(out)
    