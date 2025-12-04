
from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..layers import FNO2D

class PatchTokenizer2D(nn.Module):
    def __init__(self, in_channels: int, dim: int, patch: int = 16):
        super().__init__()
        self.patch = patch
        self.proj = nn.Conv2d(in_channels, dim, kernel_size=patch, stride=patch)
    def forward(self, x: torch.Tensor):
        x = torch.nan_to_num(x)
        x = self.proj(x)
        B, D, Hp, Wp = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # (B, Np, D)
        return tokens, (Hp, Wp)

class SpectralTokenizer2D(nn.Module):
    def __init__(self, in_channels: int, dim: int, modes: int = 16):
        super().__init__()
        self.modes = modes
        self.proj = nn.Linear(max(1, in_channels) * modes * (modes//2 + 1) * 2, dim)
    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        with torch.amp.autocast(device_type=x.device.type if hasattr(x.device,'type') else 'cuda', enabled=False):
            xf = torch.fft.rfft2(x.float(), norm='ortho')
            xf = xf[:, :, : self.modes, : self.modes//2 + 1]
            feat = torch.stack([xf.real, xf.imag], dim=-1).reshape(B, -1)
        out = self.proj(feat).unsqueeze(1)
        return torch.nan_to_num(out)

class PhysicsContextFiLM(nn.Module):
    def __init__(self, context_dim: int, dim: int):
        super().__init__()
        self.to_gb = nn.Sequential(
            nn.Linear(max(1, context_dim), dim*2), nn.SiLU(), nn.Linear(dim*2, dim*2)
        )
        self.cls = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
    def forward(self, spatial_tokens: torch.Tensor, context: Optional[torch.Tensor]):
        if context is None:
            return spatial_tokens, None
        context = torch.nan_to_num(context)
        wdtype = next(self.to_gb.parameters()).dtype
        context = context.to(dtype=wdtype, device=spatial_tokens.device)
        gb = self.to_gb(context)
        gamma, beta = gb.chunk(2, dim=-1)
        spatial_tokens = spatial_tokens * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
        cls = self.cls.expand(spatial_tokens.size(0), -1, -1)
        return torch.nan_to_num(spatial_tokens), torch.nan_to_num(cls)
