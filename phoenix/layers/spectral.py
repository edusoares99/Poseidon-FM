# phoenix/layers/spectral.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

# Optional: in case your utils has a context manager for safe FFT under AMP
try:
    from ..utils import no_autocast_if_fft
except Exception:
    from contextlib import contextmanager
    @contextmanager
    def no_autocast_if_fft(_x):
        yield

__all__ = ["LayerNorm2d", "ConvNeXtBlock", "SpectralConv2d", "FNO2D"]


class LayerNorm2d(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.var(dim=(2, 3), keepdim=True, unbiased=False)
        mean = x.mean(dim=(2, 3), keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight + self.bias


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.ln = LayerNorm2d(dim)
        hidden = int(dim * mlp_ratio)
        self.pw1 = nn.Conv2d(dim, hidden, 1)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(hidden, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dw(x)
        y = self.ln(x)
        y = self.pw2(self.act(self.pw1(y)))
        return torch.nan_to_num(x + y)


class SpectralConv2d(nn.Module):
    """Fourier layer used by FNO. Multiplies low-frequency modes in Fourier space."""
    def __init__(self, in_channels: int, out_channels: int, modes_x: int = 16, modes_y: int = 16):
        super().__init__()
        self.modes_x = int(modes_x)
        self.modes_y = int(modes_y)
        self.scale = 1.0 / (in_channels * out_channels)
        # store real/imag parts as last dim=2 for easy float32 storage
        self.weights = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, self.modes_x, self.modes_y, 2)
        )

    def compl_mul2d(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # a: (B, Cin, Mx, My) complex
        # b: (Cin, Cout, Mx, My, 2) real+imag stacked
        br, bi = b[..., 0], b[..., 1]                       # (Cin, Cout, Mx, My)
        ar, ai = a.real, a.imag                             # (B, Cin, Mx, My)
        real = torch.einsum("bixy,ioxy->boxy", ar, br) - torch.einsum("bixy,ioxy->boxy", ai, bi)
        imag = torch.einsum("bixy,ioxy->boxy", ar, bi) + torch.einsum("bixy,ioxy->boxy", ai, br)
        return torch.complex(real, imag)                    # (B, Cout, Mx, My)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, Cin, H, W = x.shape
        out_dtype = x.dtype
        with no_autocast_if_fft(x):
            x_ft = torch.fft.rfft2(x.float())               # (B, Cin, H, W//2+1)
            mx = min(self.modes_x, x_ft.size(-2))
            my = min(self.modes_y, x_ft.size(-1))

            Cout = self.weights.size(1)
            out_ft = torch.zeros(B, Cout, x_ft.size(-2), x_ft.size(-1), dtype=torch.cfloat, device=x.device)
            if mx > 0 and my > 0:
                out_ft[:, :, :mx, :my] = self.compl_mul2d(x_ft[:, :, :mx, :my], self.weights[:, :, :mx, :my, :].float())

            x = torch.fft.irfft2(out_ft, s=(H, W))          # (B, Cout, H, W)
        # cast back to original dtype if it was fp16/bf16 under autocast
        return torch.nan_to_num(x.to(out_dtype) if out_dtype.is_floating_point else x)


class FNO2D(nn.Module):
    """Simple FNO block stack."""
    def __init__(self, in_channels: int, out_channels: int, width: int = 64, modes: int = 16, depth: int = 4):
        super().__init__()
        self.fc0 = nn.Conv2d(in_channels, width, 1)
        self.blocks = nn.ModuleList([
            nn.ModuleList([
                SpectralConv2d(width, width, modes_x=modes, modes_y=modes),
                nn.Conv2d(width, width, 1),
            ]) for _ in range(int(depth))
        ])
        self.fc1 = nn.Conv2d(width, 2 * width, 1)
        self.fc2 = nn.Conv2d(2 * width, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc0(x)
        for spec, pw in self.blocks:
            y = spec(x)
            x = F.gelu(y + pw(x))
        x = F.gelu(self.fc1(x))
        return torch.nan_to_num(self.fc2(x))
