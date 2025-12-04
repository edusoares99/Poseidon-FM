# phoenix/model/phoenix.py
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Union, Tuple
from enum import Enum

from .tokens import PatchTokenizer2D, SpectralTokenizer2D, PhysicsContextFiLM
from .encoders import SpatialEncoder, SpectralEncoder
from .backbone import TransformerBackbone, MambaBackbone
from .fusion import CrossAttention


# ============================================================
# Choice enums (string-like) and registries
# ============================================================

class BackboneType(str, Enum):
    MAMBA = "mamba"
    TRANSFORMER = "transformer"

class DecoderType(str, Enum):
    FNO = "fno"
    CONV = "conv"

BACKBONES = {
    "mamba": lambda dim, depth, heads: MambaBackbone(dim, depth=depth),
    "transformer": lambda dim, depth, heads: TransformerBackbone(dim, depth=depth, heads=heads),
}

def _choice_key(x, valid: dict, *, name: str) -> str:
    """Accept Enum or string (or object with .value). Normalize to lowercase key in 'valid'."""
    if isinstance(x, Enum):
        key = str(x.value).lower()
    else:
        key = str(getattr(x, "value", x)).lower()
    if key not in valid:
        raise ValueError(f"Invalid {name}='{x}'. Valid: {list(valid.keys())}")
    return key


# ============================================================
# Config
# ============================================================

@dataclass
class PhoenixConfig:
    # core
    in_channels: int = 4         # latent_L * max_hist
    out_channels: int = 4        # latent_L
    dim: int = 256
    patch: int = 16
    modes: int = 16
    spatial_depth: int = 4
    spectral_depth: int = 2
    backbone_depth: int = 8
    heads: int = 8
    context_dim: int = 16

    # ablations / toggles
    use_film: bool = True
    use_spectral_tok: bool = True
    use_cross_attn: bool = True
    backbone_type: Union[str, BackboneType] = "mamba"   # accepts str or Enum
    decoder_type:  Union[str, DecoderType]  = "fno"     # accepts str or Enum
    use_post_1x1: bool = True
    norm_type: str = "layer"   # "layer" | "none"


# ============================================================
# PhoenixModel
# ============================================================

class PhoenixModel(nn.Module):
    def __init__(self, cfg: PhoenixConfig):
        super().__init__()
        self.cfg = cfg

        # normalize choices
        bb_key  = _choice_key(cfg.backbone_type, BACKBONES, name="backbone_type")
        dec_key = _choice_key(cfg.decoder_type,  {"fno": 1, "conv": 1}, name="decoder_type")

        # tokenizers / FiLM
        self.patch_tok = PatchTokenizer2D(cfg.in_channels, cfg.dim, cfg.patch)
        self.spec_tok  = SpectralTokenizer2D(max(1, cfg.in_channels), cfg.dim, cfg.modes) if cfg.use_spectral_tok else None
        self.context_film = PhysicsContextFiLM(cfg.context_dim, cfg.dim) if cfg.use_film else None

        # encoders
        self.spatial_enc  = SpatialEncoder(cfg.dim, cfg.spatial_depth)
        self.spectral_enc = SpectralEncoder(cfg.dim, cfg.spectral_depth) if cfg.use_spectral_tok else None

        # fusion
        self.cross1 = CrossAttention(cfg.dim, heads=cfg.heads) if (cfg.use_cross_attn and cfg.use_spectral_tok) else None
        self.cross2 = CrossAttention(cfg.dim, heads=cfg.heads) if (cfg.use_cross_attn and cfg.use_spectral_tok) else None

        # normalization
        self.norm = nn.LayerNorm(cfg.dim) if cfg.norm_type == "layer" else nn.Identity()

        # backbone
        self.backbone = BACKBONES[bb_key](cfg.dim, depth=cfg.backbone_depth, heads=cfg.heads)

        # post 1x1 prior to decoder
        self.post = nn.Conv2d(cfg.dim, cfg.in_channels, 1) if cfg.use_post_1x1 else nn.Identity()

        # decoder
        if dec_key == "fno":
            # keep import local to avoid circulars
            from ..layers.spectral import FNO2D
            self.decoder = FNO2D(cfg.in_channels, cfg.out_channels, width=64, modes=cfg.modes, depth=4)
        else:
            self.decoder = nn.Sequential(
                nn.Conv2d(cfg.in_channels, cfg.in_channels, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(cfg.in_channels, cfg.out_channels, 1),
            )

    # ------ helpers ------
    def _tokens_to_grid(self, tokens: torch.Tensor, hw: Tuple[int,int]) -> torch.Tensor:
        B, N, D = tokens.shape
        H, W = hw
        assert N == H * W, f"_tokens_to_grid: N={N} != H*W={H*W}"
        return tokens.transpose(1,2).reshape(B, D, H, W)

    # ------ forward ------
    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = torch.nan_to_num(x)

        # tokenize
        sp_tokens, hw = self.patch_tok(x)      # (B, H'*W', D)
        cls = None
        if self.context_film is not None:
            sp_tokens, cls = self.context_film(sp_tokens, context)

        spec_tokens = None
        if self.spec_tok is not None:
            spec_tokens = self.spec_tok(x)
            spec_tokens = torch.nan_to_num(spec_tokens)

        # encoders
        sp_tokens = self.spatial_enc(sp_tokens, hw)
        if self.spectral_enc is not None and spec_tokens is not None:
            spec_tokens = self.spectral_enc(spec_tokens)

        # cross-attn
        if (self.cross1 is not None) and (spec_tokens is not None):
            sp_tokens   = self.cross1(sp_tokens, spec_tokens)
            spec_tokens = self.cross2(spec_tokens, sp_tokens)

        # fuse
        parts = []
        if cls is not None: parts.append(cls)
        parts.append(sp_tokens)
        if spec_tokens is not None: parts.append(spec_tokens)
        fused = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
        fused = self.norm(fused)
        fused = self.backbone(fused)
        fused = torch.nan_to_num(fused)

        # remount spatial tokens and decode
        start = 1 if cls is not None else 0
        Ht, Wt = hw
        num_sp = Ht * Wt
        fused_spatial = fused[:, start:start+num_sp, :]
        grid_latent = self._tokens_to_grid(fused_spatial, hw)  # (B, D, H', W')

        grid_latent = self.post(grid_latent)
        up = F.interpolate(grid_latent, size=x.shape[-2:], mode='bilinear', align_corners=False)
        y = self.decoder(up)
        y = torch.nan_to_num(y)
        return y
