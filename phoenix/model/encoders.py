
from __future__ import annotations
from typing import Tuple
import torch
import torch.nn as nn
from ..layers import ConvNeXtBlock

class SpatialEncoder(nn.Module):
    def __init__(self, dim: int, depth: int = 4):
        super().__init__()
        self.blocks = nn.ModuleList([ConvNeXtBlock(dim) for _ in range(depth)])
    def forward(self, tokens: torch.Tensor, grid_hw: Tuple[int,int]):
        B, N, D = tokens.shape
        H, W = grid_hw
        x = tokens.transpose(1, 2).reshape(B, D, H, W)
        for blk in self.blocks:
            x = blk(x)
        out = x.flatten(2).transpose(1, 2)
        return torch.nan_to_num(out)

class SpectralEncoder(nn.Module):
    def __init__(self, dim: int, depth: int = 2):
        super().__init__()
        layers = []
        for _ in range(depth):
            layers += [nn.LayerNorm(dim), nn.Linear(dim, dim*4), nn.GELU(), nn.Linear(dim*4, dim)]
        self.net = nn.Sequential(*layers)
    def forward(self, spec_tokens):
        return torch.nan_to_num(self.net(spec_tokens))
