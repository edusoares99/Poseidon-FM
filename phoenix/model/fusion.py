
from __future__ import annotations
import torch
import torch.nn as nn

class CrossAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 4):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.to_q = nn.Linear(dim, dim)
        self.to_kv = nn.Linear(dim, dim*2)
        self.proj = nn.Linear(dim, dim)
    def forward(self, x_q, x_kv):
        B, Nq, D = x_q.shape
        _, Nk, _ = x_kv.shape
        q = self.to_q(x_q).reshape(B, Nq, self.heads, D//self.heads).transpose(1,2)
        kv = self.to_kv(x_kv).reshape(B, Nk, 2, self.heads, D//self.heads)
        k, v = kv[:,:,0].transpose(1,2), kv[:,:,1].transpose(1,2)
        attn = (q @ k.transpose(-2,-1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = out.transpose(1,2).reshape(B, Nq, D)
        return torch.nan_to_num(x_q + self.proj(out))
