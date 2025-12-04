
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils import no_autocast_if_fft

class LayerNorm2d(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.eps = eps
    def forward(self, x):
        var = x.var(dim=(2,3), keepdim=True, unbiased=False)
        mean = x.mean(dim=(2,3), keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight + self.bias

class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.ln = LayerNorm2d(dim)
        hidden = int(dim * mlp_ratio)
        self.pw1 = nn.Conv2d(dim, hidden, 1)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(hidden, dim, 1)
    def forward(self, x):
        x = x + self.dw(x)
        y = self.ln(x)
        y = self.pw2(self.act(self.pw1(y)))
        return torch.nan_to_num(x + y)

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes_x=16, modes_y=16):
        super().__init__()
        self.modes_x = modes_x
        self.modes_y = modes_y
        self.scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(self.scale * torch.randn(in_channels, out_channels, modes_x, modes_y, 2))
    def compl_mul2d(self, a, b):
        br, bi = b[...,0], b[...,1]
        ar, ai = a.real, a.imag
        real = torch.einsum('bixy,ioxy->boxy', ar, br) - torch.einsum('bixy,ioxy->boxy', ai, bi)
        imag = torch.einsum('bixy,ioxy->boxy', ar, bi) + torch.einsum('bixy,ioxy->boxy', ai, br)
        return torch.complex(real, imag)
    def forward(self, x):
        B, C, H, W = x.shape
        with no_autocast_if_fft(x):
            x_ft = torch.fft.rfft2(x.float())
            mx = min(self.modes_x, x_ft.size(-2)); my = min(self.modes_y, x_ft.size(-1))
            out_ft = torch.zeros(B, self.weights.size(1), x_ft.size(-2), x_ft.size(-1),
                                 dtype=torch.cfloat, device=x.device)
            out_ft[:, :, :mx, :my] = self.compl_mul2d(x_ft[:, :, :mx, :my], self.weights[:, :, :mx, :my, :].float())
            x = torch.fft.irfft2(out_ft, s=(H, W))
        return torch.nan_to_num(x.to(dtype=x.dtype) if x.dtype.is_floating_point else x)

class FNO2D(nn.Module):
    def __init__(self, in_channels, out_channels, width=64, modes=16, depth=4):
        super().__init__()
        self.fc0 = nn.Conv2d(in_channels, width, 1)
        self.blocks = nn.ModuleList([nn.ModuleList([
            SpectralConv2d(width, width, modes, modes),
            nn.Conv2d(width, width, 1)
        ]) for _ in range(depth)])
        self.fc1 = nn.Conv2d(width, 2*width, 1)
        self.fc2 = nn.Conv2d(2*width, out_channels, 1)
    def forward(self, x):
        x = self.fc0(x)
        for spec, pw in self.blocks:
            y = spec(x)
            x = F.gelu(y + pw(x))
        x = F.gelu(self.fc1(x))
        return torch.nan_to_num(self.fc2(x))
