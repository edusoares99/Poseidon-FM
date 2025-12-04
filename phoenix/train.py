
from __future__ import annotations
import os, json, math, itertools
from dataclasses import asdict
from contextlib import nullcontext
from typing import Optional, Dict, List
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from .config import PhoenixConfig
from .utils import (
    sanitize, match_ctx_dim, should_force_fp32, total_grad_norm,
    param_checksum, is_main_process, log, iter_with_retry
)
from .data import _as_btc_hw, stack_history, make_loader, well_dataset_with_retry, infer_signature
from .losses import CompositeLoss
from .model.phoenix import PhoenixModel

_PHX_CLIP_IN = float(os.environ.get("PHOENIX_CLIP_IN", "10"))

class DatasetAdapters(torch.nn.Module):
    def __init__(self, dataset_channels: Dict[str, int], max_hist: int, latent_L: int):
        super().__init__()
        self.max_hist = max_hist; self.latent_L = latent_L
        self.in_adapters = torch.nn.ModuleDict(); self.out_adapters = torch.nn.ModuleDict()
        for name, C in dataset_channels.items():
            self.in_adapters[name] = torch.nn.Conv2d(C*max_hist, latent_L*max_hist, 1)
            self.out_adapters[name] = torch.nn.Conv2d(latent_L, C, 1)
    def _pad_to_max_hist(self, x_hist: torch.Tensor, C: int, hist_eff: int) -> torch.Tensor:
        if hist_eff == self.max_hist: return x_hist
        need = C * (self.max_hist - hist_eff)
        pad = torch.zeros(x_hist.size(0), need, x_hist.size(2), x_hist.size(3),
                          device=x_hist.device, dtype=x_hist.dtype)
        return torch.cat([x_hist, pad], dim=1)
    def encode(self, name: str, x_hist: torch.Tensor, C: int, hist_eff: int) -> torch.Tensor:
        x_pad = self._pad_to_max_hist(x_hist, C, hist_eff)
        return self.in_adapters[name](x_pad)
    def decode(self, name: str, y_latent: torch.Tensor) -> torch.Tensor:
        return self.out_adapters[name](y_latent)

class MultiDataMixer:
    def __init__(self, loaders: Dict[str, torch.utils.data.DataLoader], weights: Optional[Dict[str, float]] = None, temp: float = 0.5):
        self.loaders = loaders
        self.names = list(loaders.keys())
        if weights is None:
            total = sum(len(dl.dataset) for dl in loaders.values())
            weights = {k: len(dl.dataset) / max(1, total) for k, dl in loaders.items()}
        probs = torch.tensor([weights[k] for k in self.names], dtype=torch.float32)
        probs = probs.pow(temp); probs = probs / probs.sum()
        self.probs = probs.tolist()
        self.iters = {k: iter(v) for k, v in loaders.items()}
    def __iter__(self): return self
    def __next__(self):
        import random as _r
        name = _r.choices(self.names, weights=self.probs, k=1)[0]
        try:
            batch = next(self.iters[name])
        except StopIteration:
            self.iters[name] = iter(self.loaders[name])
            batch = next(self.iters[name])
        return name, batch

class Trainer:
    def __init__(
        self,
        well_base_path: str,
        dataset_names: List[str],
        history: int = 1,
        latent_channels: int = 8,
        phoenix_dim: int = 256,
        batch_size: int = 2,
        num_workers: int = 4,
        lr: float = 2e-4,
        val_report_dir: str = "val_reports",
        cfg_overrides: Dict = None,
        loss_vrmse_w: float = 1.0,
        loss_spec_w: float = 0.2,
        loss_l1_w: float = 0.0,
        no_amp: bool = False,
        log_every: int = 200,
    ):
        self.req_history = max(1, history)
        self.latent_L = latent_channels
        self.device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0"))) if torch.cuda.is_available() else torch.device("cpu")
        self.no_amp = bool(no_amp)
        self.log_every = int(log_every)

        self.train_loaders: Dict[str, torch.utils.data.DataLoader] = {}
        self.val_loaders: Dict[str, torch.utils.data.DataLoader] = {}
        self.train_samplers: Dict[str, Optional[DistributedSampler]] = {}
        self.dataset_channels: Dict[str, int] = {}
        self.sig: Dict[str, Dict[str,int]] = {}

        base_is_hf = str(well_base_path).startswith("hf://")
        _workers = 0 if base_is_hf else num_workers

        for name in dataset_names:
            tr = well_dataset_with_retry(base=well_base_path, name=name, split="train")
            va = well_dataset_with_retry(base=well_base_path, name=name, split="valid")
            tr_sampler = DistributedSampler(tr, shuffle=True) if dist.is_initialized() else None
            va_sampler = DistributedSampler(va, shuffle=False) if dist.is_initialized() else None
            self.train_samplers[name] = tr_sampler
            self.train_loaders[name] = make_loader(tr, batch_size=batch_size, sampler=tr_sampler, num_workers=_workers, pin_memory=True)
            self.val_loaders[name] = make_loader(va, batch_size=batch_size, sampler=va_sampler, num_workers=_workers, pin_memory=True)
            sig = infer_signature(self.train_loaders[name], self.req_history)
            self.sig[name] = sig
            self.dataset_channels[name] = sig["C"]
            if is_main_process():
                log.info(f"[SIG] {name}: {sig}")

        self.max_hist = max(s['hist'] for s in self.sig.values())
        max_ctx_dim = max(s['ctx_dim'] for s in self.sig.values()) if self.sig else 16

        cfg = PhoenixConfig(
            in_channels=self.latent_L * self.max_hist,
            out_channels=self.latent_L,
            dim=phoenix_dim,
            patch=16,
            modes=12,
            context_dim=max(1, max_ctx_dim),
        )
        if cfg_overrides:
            for k, v in cfg_overrides.items():
                setattr(cfg, k, v)

        model = PhoenixModel(cfg).to(self.device)
        adapters = DatasetAdapters(self.dataset_channels, max_hist=self.max_hist, latent_L=self.latent_L).to(self.device)

        if dist.is_initialized() and self.device.type == "cuda":
            lrk = int(os.environ.get("LOCAL_RANK", "0"))
            model = DDP(model, device_ids=[lrk], output_device=lrk, find_unused_parameters=False)
            adapters = DDP(adapters, device_ids=[lrk], output_device=lrk, find_unused_parameters=False)

        self.model = model
        self.adapters = adapters
        self.optimizer = torch.optim.AdamW(
            list(self._unwrap(self.model).parameters()) + list(self._unwrap(self.adapters).parameters()),
            lr=lr, weight_decay=1e-2
        )
        self.scaler = torch.amp.GradScaler('cuda', enabled=(self.device.type == 'cuda' and not self.no_amp))
        self.mixer = MultiDataMixer(self.train_loaders)
        self.val_report_dir = val_report_dir
        self.cfg = cfg
        if is_main_process():
            os.makedirs(self.val_report_dir, exist_ok=True)

        self.criterion = CompositeLoss(loss_vrmse_w, loss_spec_w, loss_l1_w)

    def _unwrap(self, m):
        return m.module if isinstance(m, DDP) else m

    def forward_batch(self, name: str, batch):
        frames, ctx = _as_btc_hw(batch)
        frames = sanitize(frames.to(self.device, non_blocking=True), clip=_PHX_CLIP_IN)
        ctx = ctx.to(self.device, non_blocking=True) if ctx is not None else None

        hist_target = self.sig[name]["hist"]
        x, y, hist_eff, C = stack_history(frames, hist_target)  # x: (B, C*hist_eff, H, W)
        y = sanitize(y, clip=_PHX_CLIP_IN)

        ad = self._unwrap(self.adapters)
        md = self._unwrap(self.model)

        ctx = match_ctx_dim(ctx, md.cfg.context_dim)
        if ctx is not None:
            ctx = ctx.to(dtype=next(md.parameters()).dtype, device=self.device)

        use_fp32 = should_force_fp32(name)
        ac = (torch.autocast(device_type="cuda", dtype=torch.float16)
              if (not use_fp32 and self.device.type == "cuda" and not self.no_amp) else nullcontext())

        with ac:
            x_lat = ad.encode(name, x, C=C, hist_eff=hist_eff)
            x_lat = torch.nan_to_num(x_lat)
            y_lat = md(x_lat, context=ctx)
            y_lat = torch.nan_to_num(y_lat)
            y_hat = ad.decode(name, y_lat)
            y_hat = torch.nan_to_num(y_hat)

        if y_hat.shape[-2:] != y.shape[-2:]:
            y_hat = F.interpolate(y_hat, size=y.shape[-2:], mode="bilinear", align_corners=False)
        return y_hat, y

    @torch.no_grad()
    def evaluate_loader(self, name: str, vloader, max_batches: Optional[int] = None):
        s_vrmse, s_mae, s_mse, s_psnr, n = 0.0, 0.0, 0.0, 0.0, 0
        it = itertools.islice(iter_with_retry(vloader), max_batches)
        self._unwrap(self.model).eval(); self._unwrap(self.adapters).eval()
        for batch in it:
            yhat, y = self.forward_batch(name, batch)
            bs = y.size(0)
            l = self.criterion(yhat, y).item()
            mae = (yhat - y).abs().mean().item()
            mse = ((yhat - y) ** 2).mean().item()
            MAX_I = 1.0
            import math
            psnr = 10.0 * math.log10(MAX_I**2 / (mse + 1e-12))
            if dist.is_initialized():
                pack = torch.tensor([l*bs, mae*bs, mse*bs, psnr*bs, bs], device=y.device, dtype=torch.float64)
                dist.all_reduce(pack, op=dist.ReduceOp.SUM)
                s_vrmse += pack[0].item(); s_mae += pack[1].item(); s_mse += pack[2].item(); s_psnr += pack[3].item(); n += int(pack[4].item())
            else:
                s_vrmse += l * bs; s_mae += mae * bs; s_mse += mse * bs; s_psnr += psnr * bs; n += bs
        return dict(loss=s_vrmse/max(1,n), mae=s_mae/max(1,n), mse=s_mse/max(1,n), psnr=s_psnr/max(1,n), count=n)

    def train_one_epoch(self, steps_per_epoch: int, max_norm: float = 1.0):
        model = self._unwrap(self.model)
        adapters = self._unwrap(self.adapters)
        model.train(); adapters.train()
        running_loss = 0.0; last_log = 0

        model_ck = param_checksum(model); ad_ck = param_checksum(adapters)

        for step in range(steps_per_epoch):
            name, batch = next(self.mixer)
            self.optimizer.zero_grad(set_to_none=True)
            yhat, y = self.forward_batch(name, batch)
            loss = self.criterion(yhat, y)
            self.scaler.scale(loss).backward()

            self.scaler.unscale_(self.optimizer)
            gn = total_grad_norm(list(model.parameters()) + list(adapters.parameters()))
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(adapters.parameters()), max_norm)
            if (not math.isfinite(gn)) or (gn == 0.0):
                if is_main_process():
                    log.warning(f"grad-norm={gn:.3e} at step {step+1}")

            if self.scaler.is_enabled():
                old_scale = self.scaler.get_scale()
                self.scaler.step(self.optimizer); self.scaler.update()
                new_scale = self.scaler.get_scale()
                if new_scale > old_scale and is_main_process():
                    log.warning(f"AMP skipped optimizer.step() at step {step+1} (overflow). Try --no_amp or lower lr.")
            else:
                self.optimizer.step()

            loss_val = float(loss.item())
            if dist.is_initialized():
                t = torch.tensor([loss_val], device=y.device, dtype=torch.float64)
                dist.all_reduce(t, op=dist.ReduceOp.AVG)
                loss_val = float(t.item())
            running_loss += loss_val

            if is_main_process() and ((step + 1) % 200 == 0 or (step + 1) == steps_per_epoch):
                steps_done = step + 1
                window = steps_done - last_log
                avg_loss = running_loss / max(1, window)
                last_log = steps_done
                running_loss = 0.0
                lr_cur = self.optimizer.param_groups[0].get("lr", None)
                lr_str = f" | lr {lr_cur:.2e}" if lr_cur is not None else ""
                log.info(f"Step {steps_done}/{steps_per_epoch} | mix:{name} | loss {loss.item():.4f} | avg {avg_loss:.4f}{lr_str}")

        new_model_ck = param_checksum(model); new_ad_ck = param_checksum(adapters)
        if is_main_process():
            log.info(f"[debug] Δ|model|_1={new_model_ck - model_ck:.6f}  Δ|adapters|_1={new_ad_ck - ad_ck:.6f}")

    def validate_all(self):
        report = {}
        for ds_name, vloader in self.val_loaders.items():
            report[ds_name] = self.evaluate_loader(ds_name, vloader)
        if is_main_process():
            log.info("[val] " + json.dumps(report, indent=2))
            os.makedirs(self.val_report_dir, exist_ok=True)
        return report

    def save_checkpoint(self, save_dir: str, epoch: int):
        os.makedirs(save_dir, exist_ok=True)
        ckpt = {
            "epoch": epoch,
            "model": self._unwrap(self.model).state_dict(),
            "adapters": self._unwrap(self.adapters).state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "cfg": asdict(self.cfg),
        }
        path = os.path.join(save_dir, f"foundation_epoch{epoch}.pt")
        torch.save(ckpt, path)
        if is_main_process():
            log.info(f"Saved checkpoint to {path}")
