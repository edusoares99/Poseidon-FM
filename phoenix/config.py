
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum

class Backbone(str, Enum):
    mamba = "mamba"
    transformer = "transformer"

class Decoder(str, Enum):
    fno = "fno"
    conv = "conv"

class NormType(str, Enum):
    layer = "layer"
    none = "none"

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

    # ablation flags
    use_film: bool = True              # FiLM context
    use_spectral_tok: bool = True      # spectral tokenizer branch
    use_cross_attn: bool = True        # cross-attn between branches
    backbone_type: Backbone = Backbone.mamba
    decoder_type: Decoder = Decoder.fno
    use_post_1x1: bool = True          # 1x1 conv before decoder
    norm_type: NormType = NormType.layer

    # Optional: future FNO tokenizer flags could live here too
