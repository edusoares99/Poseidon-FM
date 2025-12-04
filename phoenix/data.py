
from __future__ import annotations
import os, itertools, time
from typing import Optional, Dict, Tuple
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from .utils import is_main_process
try:
    from the_well.data import WellDataset
except Exception:
    WellDataset = None  # Placeholder for environments without the_well

@torch.no_grad()
def _as_btc_hw(batch, *, prefer_keys=(
    "frames","data","states","x","inputs","input","u","field","fields",
    "image","images","video","sequence","traj","trajectories"
)):
    """Coerce a batch into (B,T,C,H,W) + optional context (B,P)."""
    import torch as _torch

    def _norm_5d(x: _torch.Tensor) -> _torch.Tensor:
        if x.ndim == 6:
            if x.shape[-1] <= 64 and x.shape[-1] < min(x.shape[2], x.shape[3], x.shape[4]):
                x = x.permute(0, 1, 5, 2, 3, 4).contiguous()
            mode = os.environ.get("PHX_VOL2D", "mean").lower()
            if mode == "center":
                z = x.shape[-1] // 2; x = x[..., z]
            elif mode == "max":
                x = x.max(dim=-1).values
            elif mode == "sum":
                x = x.sum(dim=-1)
            else:
                x = x.mean(dim=-1)
            return x
        if x.ndim == 5:
            if x.shape[-1] <= 64 and x.shape[-1] < x.shape[-2]:
                x = x.permute(0, 1, 4, 2, 3).contiguous()
            return x
        if x.ndim == 4:
            if x.shape[0] <= 8 and x.shape[1] <= 64:
                return x.unsqueeze(1)
            return x.unsqueeze(0).unsqueeze(0)
        raise ValueError(f"Tensor must be 4D, 5D or 6D, got shape {tuple(x.shape)}")

    if isinstance(batch, (list, tuple)):
        if len(batch) >= 1:
            first = batch[0]
            if isinstance(first, dict):
                batch = first
            elif _torch.is_tensor(first):
                return _norm_5d(first), None

    if isinstance(batch, dict):
        if ("input_fields" in batch and _torch.is_tensor(batch["input_fields"]) and
            "output_fields" in batch and _torch.is_tensor(batch["output_fields"])):
            x_in  = batch["input_fields"]
            x_out = batch["output_fields"]
            x_in  = _norm_5d(x_in)
            x_out = _norm_5d(x_out)
            x = _torch.cat([x_in, x_out], dim=1)
            ctx_parts = []
            for ck in ("constant_scalars","input_time_grid","output_time_grid"):
                if ck in batch and _torch.is_tensor(batch[ck]):
                    t = batch[ck]
                    if t.ndim == 1: t = t.unsqueeze(1)
                    ctx_parts.append(t.reshape(t.shape[0], -1))
            if "boundary_conditions" in batch and _torch.is_tensor(batch["boundary_conditions"]):
                bc = batch["boundary_conditions"]
                ctx_parts.append(bc.reshape(bc.shape[0], -1))
            ctx = _torch.cat(ctx_parts, dim=1) if len(ctx_parts) > 0 else None
            return x, ctx

        for k in prefer_keys:
            if k in batch and _torch.is_tensor(batch[k]):
                x = _norm_5d(batch[k])
                ctx = None
                for ck in ("context","params","physics","pde_params","cond","meta","constant_scalars"):
                    if ck in batch and _torch.is_tensor(batch[ck]):
                        ctx = batch[ck]; break
                if ctx is not None and ctx.ndim == 1:
                    ctx = ctx.unsqueeze(0)
                return x, ctx

        cand = [(n,t) for n,t in batch.items() if _torch.is_tensor(t) and t.ndim in (4,5)]
        if cand:
            _, t = max(cand, key=lambda kv: kv[1].numel())
            x = _norm_5d(t)
            ctx = None
            ctx_cands = [v for v in batch.values() if _torch.is_tensor(v) and v.ndim in (1,2) and v.numel() <= x.size(0)*256]
            if ctx_cands:
                c0 = ctx_cands[0]
                if c0.ndim == 1: c0 = c0.unsqueeze(1)
                ctx = c0
            return x, ctx

    if _torch.is_tensor(batch):
        return _norm_5d(batch), None

    raise ValueError("Cannot locate frames tensor in batch.")

def stack_history(frames: torch.Tensor, history: int):
    """Build (x,y) from frames (B,T,C,H,W) robustly."""
    B, T, C, H, W = frames.shape
    hist_eff = max(1, min(history, max(0, T - 1)))
    start = max(0, (T - 1) - hist_eff)
    end = T - 1
    if end <= start:
        x_seq = frames[:, :1]
    else:
        x_seq = frames[:, start:end]
    hist_eff = x_seq.size(1)
    y = frames[:, T-1]
    x = x_seq.reshape(B, hist_eff * C, H, W)
    return x, y, hist_eff, C

def make_loader(dataset, *, batch_size, sampler=None, num_workers=0, pin_memory=True):
    kwargs = dict(
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **kwargs)

def well_dataset_with_retry(*, base, name, split, max_retries=6, base_delay=2.0):
    if WellDataset is None:
        raise RuntimeError("the_well is not installed. Please add it to your environment.")
    try:
        from huggingface_hub.errors import HfHubHTTPError
    except Exception:
        class HfHubHTTPError(Exception): pass

    delay = base_delay
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            return WellDataset(well_base_path=base,
                               well_dataset_name=name,
                               well_split_name=split)
        except Exception as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            is_429 = ("429" in str(e)) or (status == 429) or isinstance(e, HfHubHTTPError)
            if is_429 or isinstance(e, (OSError, IOError)):
                if is_main_process():
                    print(f"[WARN] WellDataset({name},{split}) failed (try {attempt}/{max_retries}); sleep {delay:.1f}s and retry.")
                time.sleep(delay); delay *= 1.8; last_err = e; continue
            raise
    raise RuntimeError(f"Failed to open WellDataset({name},{split}) after {max_retries} retries") from last_err

def infer_signature(loader, req_history: int):
    first = next(iter(loader))
    frames, ctx = _as_btc_hw(first)
    _, T, C, H, W = frames.shape
    hist_eff = max(1, min(req_history, max(0, T-1)))
    ctx_dim = (ctx.shape[-1] if (ctx is not None and ctx.ndim >= 2) else 0)
    return dict(C=C, H=H, W=W, T=T, hist=hist_eff, ctx_dim=ctx_dim)


# ----- public/underscored compatibility aliases -----
try:
    as_btc_hw
except NameError:
    as_btc_hw = _as_btc_hw

try:
    _make_loader
except NameError:
    _make_loader = make_loader

try:
    _well_dataset_with_retry
except NameError:
    _well_dataset_with_retry = well_dataset_with_retry

# curate exports so both styles are importable
__all__ = [
    # public
    "make_loader", "as_btc_hw", "stack_history", "well_dataset_with_retry",
    "infer_signature",
    # underscored (legacy)
    "_make_loader", "_as_btc_hw", "_well_dataset_with_retry",
]