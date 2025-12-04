#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Load a VTK dataset + checkpoint and save side-by-side prediction vs ground-truth images.
Usage example:
  python tools/visualize_vtk_preds.py \
    --ckpt ./checkpoints_finetune_vtk/finetune_epoch010.pt \
    --vtk_root /data/drivaerml/run_11 \
    --vtk_glob "volume_11.vtu" \
    --vtk_fields "U,p" \
    --vtk_plane y \
    --vtk_slices "8,20,32,44,56" \
    --vtk_dims "256,256,64" \
    --outdir ./viz_samples \
    --vis_field p            # or U_mag
"""
from __future__ import annotations
import os, argparse, torch, numpy as np
import matplotlib.pyplot as plt

from phoenix.data.vtk_dataset import VTKSliceDataset
from phoenix.model.phoenix import PhoenixModel, PhoenixConfig
from phoenix.train import DatasetAdapters
from phoenix.utils import sanitize, match_ctx_dim
from dataclasses import asdict

def _cfg_from_ckpt(ckpt):
    cfg_dict = ckpt.get("cfg", None)
    if cfg_dict is None:
        raise RuntimeError("Checkpoint lacks cfg; please finetune with finetune_vtk.py to embed cfg.")
    # convert to PhoenixConfig (fields are plain)
    return PhoenixConfig(**cfg_dict)

def _build_model_adapters(ckpt, ds_channels, max_hist, device):
    cfg = _cfg_from_ckpt(ckpt)
    model = PhoenixModel(cfg).to(device)
    adapters = DatasetAdapters(ds_channels, max_hist=max_hist, latent_L=cfg.out_channels).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    adapters.load_state_dict(ckpt.get("adapters", {}), strict=False)
    model.eval(); adapters.eval()
    return model, adapters

def _stack_history(frames, hist):
    """
    frames: (B,T,C,H,W). We split into x (first hist) and y (next 1 frame).
    If T == len(slices) and you set --history=1 during train, this will use
    the first T-1 slices as input and the last slice as target (simple split).
    """
    B, T, C, H, W = frames.shape
    hist_eff = max(1, min(hist, max(0, T - 1)))
    x = frames[:, :hist_eff].reshape(B, hist_eff * C, H, W)
    y = frames[:, hist_eff:hist_eff+1].reshape(B, C, H, W)
    return x, y, hist_eff, C

def _field_channel_map(fields):
    """
    Given fields tuple like ("U","p") return mapping to channel indices:
    U splits into 3 channels (Ux,Uy,Uz), then p (1 channel), so
    returns dict: {'Ux':0, 'Uy':1, 'Uz':2, 'p':3}
    """
    ch_map = {}
    idx = 0
    for f in fields:
        if f.lower() == "u":
            ch_map["Ux"] = idx; ch_map["Uy"] = idx+1; ch_map["Uz"] = idx+2; idx += 3
        else:
            ch_map[f] = idx; idx += 1
    return ch_map

def _select_for_vis(tensor_CHW, vis_field, ch_map):
    """
    tensor_CHW: (C,H,W) numpy/tensor
    vis_field: 'p' | 'Ux' | 'Uy' | 'Uz' | 'U_mag'
    """
    if isinstance(tensor_CHW, torch.Tensor):
        arr = tensor_CHW.detach().cpu().float().numpy()
    else:
        arr = tensor_CHW
    if vis_field == "U_mag":
        cidx = [ch_map["Ux"], ch_map["Uy"], ch_map["Uz"]]
        v = np.sqrt(np.sum(arr[cidx] ** 2, axis=0))
        return v
    else:
        c = ch_map.get(vis_field, None)
        if c is None:
            raise ValueError(f"Unknown vis_field '{vis_field}'. Available: {list(ch_map.keys()) + ['U_mag']}")
        return arr[c]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--vtk_root", required=True)
    ap.add_argument("--vtk_glob", required=True)
    ap.add_argument("--vtk_fields", default="U,p")
    ap.add_argument("--vtk_plane", default="y", choices=["x","y","z"])
    ap.add_argument("--vtk_slices", default="8,20,32,44,56")
    ap.add_argument("--vtk_dims", default="256,256,64")
    ap.add_argument("--outdir", default="./viz_samples")
    ap.add_argument("--vis_field", default="p", help="Which field to visualize: p | Ux | Uy | Uz | U_mag")
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    fields = tuple(s.strip() for s in args.vtk_fields.split(",") if s.strip())
    ks = tuple(int(x) for x in args.vtk_slices.split(",") if x.strip())
    H,W,D = [int(x) for x in args.vtk_dims.split(",")]

    ds = VTKSliceDataset(args.vtk_root, args.vtk_glob, fields=fields,
                         plane=args.vtk_plane, slice_idx=ks, dims=(H,W,D))
    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    os.makedirs(args.outdir, exist_ok=True)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    # signature for adapters
    with torch.no_grad():
        sample_frames, sample_ctx = next(iter(loader))
        B,T,C,Hs,Ws = sample_frames.shape
        ds_channels = {"vtk": C}
        max_hist = max(1, min(1, max(0, T - 1)))  # assume training with history=1
        model, adapters = _build_model_adapters(ckpt, ds_channels, max_hist, device)

    ch_map = _field_channel_map(fields)

    for i, (frames, ctx) in enumerate(loader):
        frames = frames.to(device)
        ctx = ctx.to(device)
        # match model context dim
        ctx = match_ctx_dim(ctx, model.cfg.context_dim).to(device=device, dtype=next(model.parameters()).dtype)

        x, y, hist_eff, Cbase = _stack_history(frames, hist=1)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device.type=='cuda')), torch.no_grad():
            x_lat = adapters.encode("vtk", x, C=Cbase, hist_eff=hist_eff)
            y_lat = model(x_lat, context=ctx)
            y_hat = adapters.decode("vtk", y_lat)

        pred = y_hat[0]   # (C,H,W)
        gt   = y[0]       # (C,H,W)

        P = _select_for_vis(pred, args.vis_field, ch_map)
        G = _select_for_vis(gt,   args.vis_field, ch_map)

        # normalize for visualization (same scale)
        vmin = np.percentile(G, 1); vmax = np.percentile(G, 99)
        vmin, vmax = float(vmin), float(vmax)
        vmax = max(vmax, vmin + 1e-6)

        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        im0 = ax[0].imshow(G, origin="lower", vmin=vmin, vmax=vmax)
        ax[0].set_title("Ground Truth")
        ax[0].axis("off")
        im1 = ax[1].imshow(P, origin="lower", vmin=vmin, vmax=vmax)
        ax[1].set_title("Prediction")
        ax[1].axis("off")
        fig.colorbar(im1, ax=ax.ravel().tolist(), shrink=0.8)
        out = os.path.join(args.outdir, f"sample_{i:04d}_{args.vis_field}.png")
        plt.tight_layout(); plt.savefig(out, dpi=150); plt.close(fig)
        print(f"[viz] wrote {out}")

        if i >= 24:
            break  # save first 25 samples

if __name__ == "__main__":
    main()
