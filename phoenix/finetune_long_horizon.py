#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os, sys, json, math, itertools, argparse, csv
from dataclasses import asdict
from typing import Dict, Optional, Tuple, Any
from contextlib import nullcontext

import torch
import torch.distributed as dist
import torch.nn.functional as F  # only used if we need to resize outputs
from torch.nn.parallel import DistributedDataParallel as DDP

# ==============================================================
# Utils
# ==============================================================
from phoenix.utils import (
    seed_everything, ddp_setup, ddp_cleanup, is_main_process,
    sanitize, param_checksum, total_grad_norm, match_ctx_dim,
)

_sanitize = sanitize
_param_checksum = param_checksum
_total_grad_norm = total_grad_norm
_match_ctx_dim = match_ctx_dim

# ==============================================================
# Data imports (support both underscored and public names)
# ==============================================================
try:
    from phoenix.data import (
        _make_loader as make_loader,
        _as_btc_hw as as_btc_hw,
        stack_history,
        _well_dataset_with_retry as well_dataset_with_retry,
    )
except Exception:
    from phoenix.data import (
        make_loader,
        as_btc_hw,
        stack_history,
        well_dataset_with_retry,
    )

_make_loader = make_loader
_as_btc_hw = as_btc_hw
_well_dataset_with_retry = well_dataset_with_retry

# ==============================================================
# Model, Losses, and adapters/mixer
# ==============================================================
from phoenix.losses import VRMSELoss, SpectralL2, L1Loss
from phoenix.model.phoenix import PhoenixModel, PhoenixConfig
from phoenix.train import DatasetAdapters, MultiDataMixer


# ==============================================================
# JSON/pickle-safe helpers for cfg (fix for enum-like objects)
# ==============================================================
def _json_default(o: Any):
    """Default JSON serializer for enum-like objects with `.value` and numpy scalars."""
    if hasattr(o, "value"):
        return o.value
    try:
        import numpy as np
        if isinstance(o, np.generic):
            return o.item()
    except Exception:
        pass
    return str(o)


def _cfg_to_plain(cfg_obj: Any) -> Dict[str, Any]:
    """
    Convert PhoenixConfig (possibly containing enum-like objects) into a
    plain dict that is safe to JSON-serialize and pickle.
    """
    d = asdict(cfg_obj)

    def _map(v):
        if isinstance(v, dict):
            return {k: _map(vv) for k, vv in v.items()}
        if isinstance(v, (list, tuple)):
            return type(v)(_map(x) for x in v)
        if hasattr(v, "value"):
            return v.value
        try:
            import numpy as np
            if isinstance(v, np.generic):
                return v.item()
        except Exception:
            pass
        return v

    return _map(d)


# ==============================================================
# Helpers
# ==============================================================
def _parse_bins(bins_str: str):
    """
    "1,2:8,9:26,27:56" -> [('1',[1]), ('2:8',[2..8]), ('9:26',[9..26]), ('27:56',[27..56])]
    """
    bins = []
    for tok in bins_str.split(","):
        tok = tok.strip()
        if ":" in tok:
            a, b = tok.split(":")
            bins.append((tok, list(range(int(a), int(b) + 1))))
        else:
            t = int(tok)
            bins.append((tok, [t]))
    return bins


# ==============================================================
# CLI
# ==============================================================
def build_parser():
    p = argparse.ArgumentParser("phoenix.finetune")
    # data
    p.add_argument("--base", required=True, type=str, help="The Well base path or hf://datasets/... root")
    p.add_argument("--datasets", type=str, default="", help="Comma-separated list with the NEW domain(s)")
    p.add_argument("--dataset", type=str, default="", help="Single dataset shorthand")
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--workers", type=int, default=0)

    # training schedule
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--steps_per_epoch", type=int, default=800)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--history", type=int, default=1)

    # model (used if checkpoint doesn't carry cfg)
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--latent", type=int, default=8)
    p.add_argument("--patch", type=int, default=16)
    p.add_argument("--modes", type=int, default=12)

    # finetune knobs
    p.add_argument("--finetune_mode", type=str, default="adapters",
                   choices=["adapters", "adapters_decoder", "full"])
    p.add_argument("--unfreeze_last_n", type=int, default=2, help="Backbone blocks to unfreeze in 'full' mode")

    # discriminative LRs
    p.add_argument("--lr_adapters", type=float, default=3e-4)
    p.add_argument("--lr_decoder",  type=float, default=3e-4)
    p.add_argument("--lr_backbone", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)

    # losses
    p.add_argument("--loss_vrmse_w", type=float, default=1.0)
    p.add_argument("--loss_spec_w",  type=float, default=0.0)
    p.add_argument("--loss_l1_w",    type=float, default=0.0)

    # misc
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--log_every", type=int, default=200)
    p.add_argument("--save_dir", type=str, default="./checkpoints_finetune")
    p.add_argument("--checkpoint", type=str, default="", help="Foundation checkpoint (.pt)")
    p.add_argument("--force_fp32_datasets", type=str, default=os.environ.get("FORCE_FP32_DATASETS",""),
                   help="Comma list of dataset names forced to run in fp32")

    # --- NEW: PhysiX-style rollout evaluation ---
    p.add_argument("--eval_rollout_steps", type=int, default=0,
                   help="If >0, run autoregressive eval to this Δt (e.g., 56).")
    p.add_argument("--eval_bins", type=str, default="1,2:8,9:26,27:56",
                   help='Comma list of horizon bins (e.g., PhysiX Table 4).')
    p.add_argument("--eval_csv", type=str, default="",
                   help="If set, write CSV with bin VRMSE per dataset/epoch (filename auto).")
    p.add_argument("--save_best_by_long", action="store_true",
                   help="Save extra checkpoint when long-horizon bin improves.")
    return p


# ==============================================================
# FineTune Trainer
# ==============================================================
class FineTuneTrainer:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        base_is_hf = str(args.base).startswith("hf://")
        workers = 0 if base_is_hf else args.workers

        # discover dataset list
        names = []
        if args.datasets:
            names = [x.strip() for x in args.datasets.split(",") if x.strip()]
        elif args.dataset:
            names = [args.dataset]
        else:
            raise ValueError("Provide --datasets or --dataset")

        # device / DDP
        self.device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0"))) if torch.cuda.is_available() else torch.device("cpu")

        # Build loaders and gather signatures
        self.train_loaders: Dict[str, torch.utils.data.DataLoader] = {}
        self.val_loaders: Dict[str, torch.utils.data.DataLoader] = {}
        self.sig: Dict[str, Dict] = {}

        for name in names:
            tr = _well_dataset_with_retry(base=args.base, name=name, split="train")
            va = _well_dataset_with_retry(base=args.base, name=name, split="valid")

            self.train_loaders[name] = _make_loader(tr, batch_size=args.batch, sampler=None, num_workers=workers, pin_memory=True)
            self.val_loaders[name]   = _make_loader(va, batch_size=args.batch, sampler=None, num_workers=workers, pin_memory=True)

            # signature from one batch
            first = next(iter(self.train_loaders[name]))
            frames, ctx = _as_btc_hw(first)
            _, T, C, H, W = frames.shape
            hist_eff = max(1, min(args.history, max(0, T - 1)))
            ctx_dim = (ctx.shape[-1] if (ctx is not None and ctx.ndim >= 2) else 0)
            self.sig[name] = dict(C=C, H=H, W=W, T=T, hist=hist_eff, ctx_dim=ctx_dim)
            if is_main_process():
                print(f"[SIG] {name}: {self.sig[name]}")

        # model cfg + build
        self.max_hist = max(s['hist'] for s in self.sig.values())
        ds_channels = {k: s['C'] for k, s in self.sig.items()}
        max_ctx_dim = max(s['ctx_dim'] for s in self.sig.values()) if self.sig else 16

        cfg = self._load_config(args, max_ctx_dim)
        self.model = PhoenixModel(cfg).to(self.device)
        self.adapters = DatasetAdapters(ds_channels, max_hist=self.max_hist, latent_L=cfg.out_channels).to(self.device)

        # (DDP)
        if dist.is_initialized() and self.device.type == "cuda":
            lrk = int(os.environ.get("LOCAL_RANK", "0"))
            self.model = DDP(self.model, device_ids=[lrk], output_device=lrk, find_unused_parameters=False)
            self.adapters = DDP(self.adapters, device_ids=[lrk], output_device=lrk, find_unused_parameters=False)

        # Load checkpoint (foundation) safely
        if args.checkpoint:
            sd = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
            self._unwrap(self.model).load_state_dict(sd.get("model", sd), strict=False)
            self._unwrap(self.adapters).load_state_dict(sd.get("adapters", {}), strict=False)
            if is_main_process():
                print(f"[INFO] Loaded foundation checkpoint: {args.checkpoint}")

        # losses and scaler
        self.vrmse = VRMSELoss()
        self.specL = SpectralL2(1.0) if args.loss_spec_w > 0 else None
        self.l1L   = L1Loss() if args.loss_l1_w > 0 else None
        self.scaler = torch.amp.GradScaler('cuda', enabled=(self.device.type == "cuda" and not args.no_amp))

        # mixer
        self.mixer = MultiDataMixer(self.train_loaders)

        # Save dir + config dump (JSON-safe)
        if is_main_process():
            os.makedirs(args.save_dir, exist_ok=True)
            cfg_plain = _cfg_to_plain(self._unwrap(self.model).cfg)
            with open(os.path.join(args.save_dir, "finetune_config.json"), "w") as f:
                json.dump({"cfg": cfg_plain, "args": vars(args)}, f, indent=2, default=_json_default)

        # finetune mode + optimizer
        target_ds = next(iter(self.train_loaders.keys()))
        self._set_finetune_mode(args.finetune_mode, target_ds)
        self._build_optimizer()

        # fp32 mask + logging pace
        self.force_fp32 = set([x.strip() for x in args.force_fp32_datasets.split(",") if x.strip()])
        self.log_every = self.args.log_every

    def _unwrap(self, m):
        return m.module if isinstance(m, DDP) else m

    # ----------------------------------------------------------
    # Config loader that tolerates old checkpoints (str enums)
    # ----------------------------------------------------------
    def _load_config(self, args, max_ctx_dim):
        def _ensure_enum_like(v: object, *, default: str) -> object:
            """Return an object with a .value attribute (for legacy string cfgs)."""
            if hasattr(v, "value"):   # already enum-like
                return v
            if isinstance(v, str) and v:
                class _E: pass
                e = _E()
                e.value = v.lower()
                return e
            class _E: pass
            e = _E()
            e.value = default
            return e

        cfg_dict = None
        if args.checkpoint and os.path.isfile(args.checkpoint):
            ck = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
            cfg_dict = ck.get("cfg", None)

        if cfg_dict is None:
            # fresh finetune config (minimal)
            return PhoenixConfig(
                in_channels=args.latent * args.history,
                out_channels=args.latent,
                dim=args.dim,
                patch=args.patch,
                modes=args.modes,
                context_dim=max_ctx_dim,
            )

        # normalize legacy string enums
        if "backbone_type" in cfg_dict:
            cfg_dict["backbone_type"] = _ensure_enum_like(cfg_dict["backbone_type"], default="transformer")
        if "decoder_type" in cfg_dict:
            cfg_dict["decoder_type"]  = _ensure_enum_like(cfg_dict["decoder_type"],  default="fno")

        # ensure context can handle new datasets
        cfg_dict["context_dim"] = max(int(cfg_dict.get("context_dim", 1)), max_ctx_dim)

        return PhoenixConfig(**cfg_dict)

    # ----------------------------------------------------------
    # Finetune modes
    # ----------------------------------------------------------
    def _set_finetune_mode(self, mode: str, new_ds: str):
        m = self._unwrap(self.model)
        a = self._unwrap(self.adapters)

        # freeze all
        for p in m.parameters(): p.requires_grad = False
        for p in a.parameters(): p.requires_grad = False

        # adapters always on (only selected ds actually has requires_grad=True here)
        for p in a.in_adapters[new_ds].parameters():  p.requires_grad = True
        for p in a.out_adapters[new_ds].parameters(): p.requires_grad = True

        if mode in ("adapters_decoder", "full"):
            for p in m.decoder.parameters(): p.requires_grad = True
            if hasattr(m, "post") and not isinstance(m.post, torch.nn.Identity):
                for p in m.post.parameters(): p.requires_grad = True

        if mode == "full":
            N = max(1, self.args.unfreeze_last_n)
            # MambaBackbone
            if hasattr(m.backbone, "layers"):
                for layer in m.backbone.layers[-N:]:
                    for p in layer.parameters(): p.requires_grad = True
            # TransformerBackbone
            if hasattr(m.backbone, "enc"):
                for layer in m.backbone.enc.layers[-N:]:
                    for p in layer.parameters(): p.requires_grad = True

        if is_main_process():
            total = sum(p.numel() for p in m.parameters()) + sum(p.numel() for p in a.parameters())
            trainable = sum(p.requires_grad for p in itertools.chain(m.parameters(), a.parameters()))
            print(f"[INFO] Finetune mode: {mode} | trainable params set: {trainable}/{total}")

    # ----------------------------------------------------------
    # Optimizer with discriminative LRs
    # ----------------------------------------------------------
    def _build_optimizer(self):
        m = self._unwrap(self.model)
        a = self._unwrap(self.adapters)

        groups = []

        def add_group(obj, lr):
            if obj is None: return
            params = [p for p in obj.parameters() if p.requires_grad]
            if params:
                groups.append({"params": params, "lr": lr, "weight_decay": self.args.weight_decay})

        # adapters (all; only the target ds adapters have requires_grad=True)
        add_group(a, self.args.lr_adapters)

        # decoder/post if enabled
        if any(p.requires_grad for p in m.decoder.parameters()):
            add_group(m.decoder, self.args.lr_decoder)
        if hasattr(m, "post") and any(p.requires_grad for p in m.post.parameters()):
            add_group(m.post, self.args.lr_decoder)

        # backbone last layers (if any unfrozen)
        if hasattr(m.backbone, "layers"):
            last = [l for l in m.backbone.layers if any(p.requires_grad for p in l.parameters())]
            if last:
                add_group(torch.nn.ModuleList(last), self.args.lr_backbone)
        elif hasattr(m.backbone, "enc"):
            last = [l for l in m.backbone.enc.layers if any(p.requires_grad for p in l.parameters())]
            if last:
                add_group(torch.nn.ModuleList(last), self.args.lr_backbone)

        if not groups:
            raise RuntimeError("No parameters marked as trainable. Check finetune mode setup.")

        self.optimizer = torch.optim.AdamW(groups, weight_decay=self.args.weight_decay)

    # ----------------------------------------------------------
    # Loss
    # ----------------------------------------------------------
    def _compose_loss(self, yhat, y):
        loss = 0.0
        if self.args.loss_vrmse_w:
            loss = loss + self.args.loss_vrmse_w * self.vrmse(yhat, y)
        if (self.specL is not None) and self.args.loss_spec_w:
            loss = loss + self.args.loss_spec_w * self.specL(yhat, y)
        if (self.l1L is not None) and self.args.loss_l1_w:
            loss = loss + self.args.loss_l1_w * self.l1L(yhat, y)
        return loss

    # ----------------------------------------------------------
    # One forward batch (single-step)
    # ----------------------------------------------------------
    def forward_batch(self, name: str, batch):
        frames, ctx = _as_btc_hw(batch)
        frames = _sanitize(frames.to(self.device, non_blocking=True), clip=float(os.environ.get("PHOENIX_CLIP_IN", "10")))
        ctx = ctx.to(self.device, non_blocking=True) if ctx is not None else None

        hist_target = self.sig[name]["hist"]
        x, y, hist_eff, C = stack_history(frames, hist_target)
        y = _sanitize(y, clip=float(os.environ.get("PHOENIX_CLIP_OUT", "10")))

        ad = self._unwrap(self.adapters)
        base_model = self._unwrap(self.model)

        # match/pad context to model cfg
        ctx = _match_ctx_dim(ctx, base_model.cfg.context_dim)
        if ctx is not None:
            ctx = ctx.to(dtype=next(base_model.parameters()).dtype, device=self.device)

        use_fp32 = (name in self.force_fp32) or (self.device.type != "cuda") or self.args.no_amp
        ac = nullcontext() if use_fp32 else torch.autocast(device_type="cuda", dtype=torch.float16)

        with ac:
            x_lat = ad.encode(name, x, C=C, hist_eff=hist_eff)       # (B, L*max_hist, H, W)
            y_lat = self.model(x_lat, context=ctx)                   # (B, L, H, W)
            y_hat = ad.decode(name, y_lat)                           # (B, C, H, W)

        if y_hat.shape[-2:] != y.shape[-2:]:
            y_hat = F.interpolate(y_hat, size=y.shape[-2:], mode="bilinear", align_corners=False)

        return y_hat, y

    # ----------------------------------------------------------
    # Train one epoch
    # ----------------------------------------------------------
    def train_one_epoch(self, steps_per_epoch: int, epoch: int):
        model = self._unwrap(self.model)
        adapters = self._unwrap(self.adapters)

        model.train(); adapters.train()
        running_loss = 0.0
        last_log = 0

        m_ck = _param_checksum(model); a_ck = _param_checksum(adapters)

        for step in range(steps_per_epoch):
            name, batch = next(self.mixer)

            self.optimizer.zero_grad(set_to_none=True)
            yhat, y = self.forward_batch(name, batch)
            loss = self._compose_loss(yhat, y)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)

            gn = _total_grad_norm(list(model.parameters()) + list(adapters.parameters()))
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(adapters.parameters()), 1.0)

            old_scale = self.scaler.get_scale() if self.scaler.is_enabled() else None
            self.scaler.step(self.optimizer); self.scaler.update()
            if self.scaler.is_enabled():
                new_scale = self.scaler.get_scale()
                if new_scale > (old_scale or new_scale) and is_main_process():
                    print(f"[WARN] AMP overflow skipped optimizer.step() at epoch {epoch} step {step+1}")

            loss_val = float(loss.item())
            running_loss += loss_val

            if is_main_process() and ((step + 1) % self.log_every == 0 or (step + 1) == steps_per_epoch):
                window = (step + 1) - last_log
                avg_loss = running_loss / max(1, window)
                last_log = step + 1
                running_loss = 0.0
                print(f"Epoch {epoch} Step {step+1}/{steps_per_epoch} | mix:{name} | loss {loss_val:.4f} | avg {avg_loss:.4f} | grad {gn:.3e}")

        if is_main_process():
            nm_ck = _param_checksum(model); na_ck = _param_checksum(adapters)
            print(f"[debug] Δ|model|_1={nm_ck - m_ck:.6f}  Δ|adapters|_1={na_ck - a_ck:.6f}")

    # ----------------------------------------------------------
    # Validate (single-step)
    # ----------------------------------------------------------
    @torch.no_grad()
    def validate(self, max_batches: Optional[int] = 10):
        model = self._unwrap(self.model); adapters = self._unwrap(self.adapters)
        model.eval(); adapters.eval()

        for ds_name, vloader in self.val_loaders.items():
            s_vrmse = n = 0
            it = itertools.islice(iter(vloader), max_batches)
            for batch in it:
                yhat, y = self.forward_batch(ds_name, batch)
                s_vrmse += float(self.vrmse(yhat, y).item()) * y.size(0)
                n += y.size(0)
            if is_main_process():
                print(f"[VAL] {ds_name}: VRMSE={s_vrmse/max(1,n):.4f} on {n} frames")

    # ----------------------------------------------------------
    # PhysiX-style autoregressive rollout evaluation
    # ----------------------------------------------------------
    @torch.no_grad()
    def evaluate_rollout(self, ds_name: str, max_batches: int = 10):
        """
        Autoregressive rollout to Δt = args.eval_rollout_steps.
        Returns: dict { 'per_dt': {1: vrmse, ...}, 'bins': { '1': x, '2:8': y, ... } }
        """
        if self.args.eval_rollout_steps <= 0:
            return {"per_dt": {}, "bins": {}}

        device = self.device
        vloader = self.val_loaders[ds_name]
        sig = self.sig[ds_name]
        hist = sig["hist"]
        C, H, W = sig["C"], sig["H"], sig["W"]
        max_dt = self.args.eval_rollout_steps

        m = self._unwrap(self.model).eval()
        ad = self._unwrap(self.adapters).eval()

        per_dt_sum = [0.0 for _ in range(max_dt)]
        per_dt_cnt = [0 for _ in range(max_dt)]

        def _vrmse(pred, tgt):
            diff = pred - tgt
            mu = diff.mean(dim=tuple(range(1, diff.ndim)), keepdim=True)
            return torch.sqrt(((diff - mu)**2).mean()).item()

        it = itertools.islice(iter(vloader), max_batches)
        for batch in it:
            frames, ctx = _as_btc_hw(batch)           # [B, T, C, H, W]
            frames = _sanitize(frames.to(device, non_blocking=True),
                               clip=float(os.environ.get("PHOENIX_CLIP_IN", "10")))
            T = frames.shape[1]
            max_dt_eff = min(max_dt, T - hist)
            if max_dt_eff <= 0:
                continue

            base_model = self._unwrap(self.model)
            ctx = _match_ctx_dim(ctx, base_model.cfg.context_dim)
            if ctx is not None:
                ctx = ctx.to(dtype=next(base_model.parameters()).dtype, device=device)

            buf = frames[:, :hist]                     # [B, hist, C, H, W]

            for t in range(1, max_dt_eff + 1):
                Bsz = buf.size(0)
                x = buf.permute(0, 2, 1, 3, 4).reshape(Bsz, C * hist, H, W).contiguous()

                with torch.autocast(device_type="cuda", dtype=torch.float16,
                                    enabled=(self.device.type == "cuda" and not self.args.no_amp)):
                    x_lat = ad.encode(ds_name, x, C=C, hist_eff=hist)
                    y_lat = m(x_lat, context=ctx)             # (B, L, H, W)
                    y_hat = ad.decode(ds_name, y_lat)         # (B, C, H, W)

                tgt_t = frames[:, hist + t - 1]               # (B, C, H, W)
                score = _vrmse(y_hat, tgt_t)
                per_dt_sum[t - 1] += score * Bsz
                per_dt_cnt[t - 1] += Bsz

                buf = torch.cat([buf, y_hat.unsqueeze(1)], dim=1)[:, -hist:]

        per_dt = {dt: (per_dt_sum[dt - 1] / max(1, per_dt_cnt[dt - 1]))
                  for dt in range(1, max_dt + 1) if per_dt_cnt[dt - 1] > 0}

        bins = {}
        for name, dts in _parse_bins(self.args.eval_bins):
            vals = [per_dt.get(dt) for dt in dts if dt in per_dt]
            if vals:
                bins[name] = float(sum(vals) / len(vals))

        return {"per_dt": per_dt, "bins": bins}

    # ----------------------------------------------------------
    # Save checkpoint
    # ----------------------------------------------------------
    def save(self, epoch: int):
        if not is_main_process(): return
        cfg_plain = _cfg_to_plain(self._unwrap(self.model).cfg)
        ckpt = {
            "epoch": epoch,
            "model": self._unwrap(self.model).state_dict(),
            "adapters": self._unwrap(self.adapters).state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "cfg": cfg_plain,
        }
        path = os.path.join(self.args.save_dir, f"finetune_epoch{epoch:03d}.pt")
        torch.save(ckpt, path)
        print(f"[INFO] Saved: {path}")


# ==============================================================
# Entry point
# ==============================================================
def main():
    ddp_setup()
    try:
        args = build_parser().parse_args()
        seed_everything(42)
        trainer = FineTuneTrainer(args)

        parsed_bins = _parse_bins(args.eval_bins)  # trigger early parse errors

        for epoch in range(1, args.epochs + 1):
            trainer.train_one_epoch(args.steps_per_epoch, epoch)
            trainer.validate(max_batches=10)

            # === PhysiX-style rollout eval ===
            if args.eval_rollout_steps > 0:
                for ds_name in trainer.val_loaders.keys():
                    out = trainer.evaluate_rollout(ds_name, max_batches=10)
                    if is_main_process():
                        bins = out["bins"]
                        if bins:
                            pretty = " | ".join(f"{k}={v:.3f}" for k, v in bins.items())
                            print(f"[EVAL-ROLLOUT] {ds_name} Δt-bins: {pretty}")

                        # save CSV if requested
                        if args.eval_csv:
                            os.makedirs(args.save_dir, exist_ok=True)
                            csv_path = os.path.join(args.save_dir, f"eval_bins_{ds_name}_epoch{epoch:03d}.csv")
                            with open(csv_path, "w", newline="") as f:
                                w = csv.writer(f); w.writerow(["bin", "vrmse"])
                                for k, v in bins.items():
                                    w.writerow([k, f"{v:.6f}"])
                            print(f"[INFO] wrote {csv_path}")

                        # optional: best-by-longest bin (last bin in list)
                        if args.save_best_by_long and bins:
                            long_name = parsed_bins[-1][0]
                            long_val = bins.get(long_name, None)
                            if long_val is not None:
                                best_file = os.path.join(args.save_dir, f"best_{ds_name}_long.pt")
                                state = {"epoch": epoch,
                                         "model": trainer._unwrap(trainer.model).state_dict(),
                                         "adapters": trainer._unwrap(trainer.adapters).state_dict(),
                                         "cfg": _cfg_to_plain(trainer._unwrap(trainer.model).cfg),
                                         "long_bin": long_name, "long_vrmse": float(long_val)}
                                save_it = True
                                if os.path.exists(best_file):
                                    try:
                                        prev = torch.load(best_file, map_location="cpu")
                                        save_it = (float(long_val) < float(prev.get("long_vrmse", 1e9)) - 1e-6)
                                    except Exception:
                                        pass
                                if save_it:
                                    torch.save(state, best_file)
                                    print(f"[INFO] Saved best-long checkpoint: {best_file} (Δt {long_name} = {long_val:.3f})")

            trainer.save(epoch)

        if is_main_process():
            print("✅ Finetuning complete.")
    finally:
        ddp_cleanup()


if __name__ == "__main__":
    main()
