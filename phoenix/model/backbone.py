
from __future__ import annotations
import torch
import torch.nn as nn

BACKBONES = {}

def register_backbone(name):
    def wrap(cls):
        BACKBONES[name] = cls
        return cls
    return wrap

@register_backbone("transformer")
class TransformerBackbone(nn.Module):
    def __init__(self, dim: int, depth: int = 6, heads: int = 8, **kwargs):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=dim*4, batch_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=depth)
    def forward(self, x):
        return torch.nan_to_num(self.enc(x))

@register_backbone("mamba")
class MambaBackbone(nn.Module):
    def __init__(self, dim: int, depth: int = 8, **kwargs):
        super().__init__()
        try:
            from mamba_ssm import Mamba
            self.layers = nn.ModuleList([Mamba(d_model=dim, d_state=16, d_conv=4, expand=2) for _ in range(depth)])
            self.use_mamba = True
        except Exception:
            enc_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=8, dim_feedforward=dim*4, batch_first=True)
            self.enc = nn.TransformerEncoder(enc_layer, num_layers=depth)
            self.use_mamba = False
    def forward(self, x):
        if getattr(self, 'use_mamba', False):
            for layer in self.layers:
                x = x + layer(x)
            return torch.nan_to_num(x)
        return torch.nan_to_num(self.enc(x))
