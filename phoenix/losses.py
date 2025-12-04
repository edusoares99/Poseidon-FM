
from __future__ import annotations
import torch
import torch.nn as nn

class VRMSELoss(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__(); self.eps = eps
    def forward(self, pred, target):
        err = pred - target
        err = err - err.mean(dim=(2,3), keepdim=True)
        mse = (err * err).mean()
        return torch.sqrt(mse + self.eps)

class SpectralL2(nn.Module):
    def __init__(self, weight_high: float = 1.0):
        super().__init__(); self.weight_high = weight_high
    def forward(self, pred, target):
        with torch.amp.autocast(device_type=pred.device.type if hasattr(pred.device,'type') else 'cuda', enabled=False):
            pf = torch.fft.rfft2(pred.float(), norm='ortho')
            tf = torch.fft.rfft2(target.float(), norm='ortho')
            diff = pf - tf
            H, W2 = diff.shape[-2], diff.shape[-1]
            device = diff.device
            ky = torch.fft.fftfreq(H, d=1.0).to(device)[:, None]
            kx = torch.fft.rfftfreq((W2 - 1) * 2, d=1.0).to(device)[None, :]
            k = torch.sqrt(ky**2 + kx**2)
            w = 1.0 + self.weight_high * k / (k.max() + 1e-8)
            w = w[None, None, :, :]
            loss = (w * (diff.real**2 + diff.imag**2)).mean()
        return loss

class L1Loss(nn.Module):
    def forward(self, pred, target):
        return (pred - target).abs().mean()

class CompositeLoss(nn.Module):
    def __init__(self, vrmse_w: float = 1.0, spec_w: float = 0.2, l1_w: float = 0.0):
        super().__init__()
        self.vrmse = VRMSELoss()
        self.spec  = SpectralL2(1.0)
        self.l1    = L1Loss()
        self.vrmse_w = float(vrmse_w)
        self.spec_w  = float(spec_w)
        self.l1_w    = float(l1_w)
    def forward(self, yhat, y):
        loss = 0.0
        if self.vrmse_w: loss = loss + self.vrmse_w * self.vrmse(yhat, y)
        if self.spec_w:  loss = loss + self.spec_w  * self.spec(yhat, y)
        if self.l1_w:    loss = loss + self.l1_w    * self.l1(yhat, y)
        return loss
