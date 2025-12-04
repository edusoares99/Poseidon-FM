
from __future__ import annotations
import os, math, random, time, logging
from contextlib import contextmanager, nullcontext
from typing import Optional, Iterable
import torch
import torch.distributed as dist

def get_logger(name: str = "phoenix"):
    level = os.environ.get("PHX_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(message)s",
        level=getattr(logging, level, logging.INFO),
    )
    return logging.getLogger(name)

log = get_logger()

def ddp_setup():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ and not dist.is_initialized():
        dist.init_process_group(backend='nccl', init_method='env://')
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        if torch.cuda.device_count() == 0:
            raise RuntimeError("No CUDA devices visible.")
        if local_rank >= torch.cuda.device_count():
            raise RuntimeError(f"LOCAL_RANK={local_rank} but {torch.cuda.device_count()} CUDA devices visible.")
        torch.cuda.set_device(local_rank)
        try:
            dist.barrier(device_ids=[torch.cuda.current_device()])
        except Exception:
            dist.barrier()

def ddp_cleanup():
    try:
        if dist.is_available() and dist.is_initialized():
            try: dist.barrier()
            except Exception: pass
            try: dist.destroy_process_group()
            except Exception as e: log.warning(f"destroy_process_group failed: {e}")
    finally:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

def get_rank():
    return dist.get_rank() if (dist.is_available() and dist.is_initialized()) else 0

def is_main_process():
    return get_rank() == 0

def seed_everything(seed: int = 42):
    import numpy as np
    pyr = random
    rank = get_rank()
    s = seed + rank
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    np.random.seed(s)
    pyr.seed(s)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def sanitize(t: torch.Tensor, clip: float = 10.0) -> torch.Tensor:
    if t.dtype == torch.float64:
        t = t.to(torch.float32)
    t = torch.nan_to_num(t, nan=0.0, posinf=clip, neginf=-clip)
    if clip is not None:
        t = t.clamp(-clip, clip)
    return t

def total_grad_norm(params: Iterable[torch.nn.Parameter], norm_type=2.0) -> float:
    norms = []
    for p in params:
        if p.grad is not None:
            norms.append(p.grad.data.detach().float().norm(norm_type))
    if not norms:
        return 0.0
    return float(torch.stack(norms).norm(norm_type).item())

def param_checksum(module: torch.nn.Module) -> float:
    s = 0.0
    with torch.no_grad():
        for p in module.parameters():
            if p.dtype.is_floating_point and p.numel() > 0:
                s += float(p.detach().double().abs().sum().item())
    return s

def should_force_fp32(dataset_name: str) -> bool:
    forced = os.environ.get("FORCE_FP32_DATASETS", "")
    if not forced: return False
    names = [s.strip() for s in forced.split(",") if s.strip()]
    return dataset_name in names

@contextmanager
def no_autocast_if_fft(x: torch.Tensor):
    dev_type = x.device.type if hasattr(x.device, "type") else "cuda"
    with torch.amp.autocast(device_type=dev_type, enabled=False):
        yield

def match_ctx_dim(ctx: Optional[torch.Tensor], target_dim: int) -> Optional[torch.Tensor]:
    if ctx is None or target_dim is None or target_dim <= 0:
        return None if ctx is None else ctx
    if ctx.ndim == 1:
        ctx = ctx.unsqueeze(0)
    if ctx.ndim > 2:
        b = ctx.shape[0]
        ctx = ctx.reshape(b, -1)
    P = ctx.shape[-1]
    if P == target_dim: return ctx
    if P > target_dim: return ctx[..., :target_dim]
    pad = target_dim - P
    return torch.nn.functional.pad(ctx, (0, pad), mode='constant', value=0.0)

def iter_with_retry(dataloader, *, max_retries=6, base_delay=2.0):
    """Yield batches with retry on transient I/O/HF429 errors."""
    import time as _time, random as _rand
    try:
        from huggingface_hub.errors import HfHubHTTPError
    except Exception:
        class HfHubHTTPError(Exception): pass
    it = iter(dataloader)
    while True:
        try:
            yield next(it)
        except StopIteration:
            return
        except Exception as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            is_429 = (status == 429) or ("429" in str(e)) or isinstance(e, HfHubHTTPError)
            transient = is_429 or isinstance(e, (OSError, IOError))
            if not transient:
                raise
            for attempt in range(max_retries):
                delay = base_delay * (2 ** attempt) + _rand.random()
                if is_main_process():
                    log.warning(f"Data fetch error ({'429' if is_429 else type(e).__name__}); retry {attempt+1}/{max_retries} after {delay:.1f}s")
                _time.sleep(delay)
                try:
                    it = iter(dataloader)
                    break
                except Exception:
                    pass
            else:
                raise RuntimeError("Exceeded maximum retries while fetching data") from e
