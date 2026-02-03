# assembles entire transformer architecture

from torch import nn
from .TransformerBlock import TransformerBlock
from .PositionalEncoding import PositionalEncoding
import torch
from PatchEmbedding import PatchEmbedding_CNN, PatchEmbedding_Linear

class SimpleTransformer(nn.Module):
    def __init__(self, dim, num_unique_tokens=10, num_layers=6, heads=8, dim_head=None, max_seq_len=1024):
        super().__init__()
        self.max_seq_len = max_seq_len

        # Patch embedding (choose one implementation)
        self.token_emb = PatchEmbedding_CNN(emb_size=dim)
        # self.token_emb = PatchEmbedding_Linear(emb_size=dim)

        self.pos_enc = PositionalEncoding(dim, max_seq_length=max_seq_len)

        self.block_list = [TransformerBlock(dim=dim, heads=heads, dim_head=dim_head) for _ in range(num_layers)]
        self.layers = nn.ModuleList(self.block_list)
        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_unique_tokens)
        )
    def forward(self, x, mask  = None): # x shape: (batch_size, channels, height, width)
        pos = torch.arange(0, x.shape[-1], dtype=torch.long, device=x.device).unsqueeze(0)
        x = self.token_emb(x)
        x = x + self.pos_enc(x)
        for layer in self.layers:
            x = layer(x)
        return self.to_logits(x)
