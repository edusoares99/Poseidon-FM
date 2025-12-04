
from __future__ import annotations
import os, sys, json, argparse
import torch
import torch.distributed as dist
from .utils import ddp_setup, ddp_cleanup, seed_everything, is_main_process, log
from .config import PhoenixConfig, Backbone, Decoder, NormType
from .train import Trainer

def exp_run_id(args, cfg: PhoenixConfig):
    bits = [
        f"bb={cfg.backbone_type.value}",
        f"dec={cfg.decoder_type.value}",
        f"film={int(cfg.use_film)}",
        f"specTok={int(cfg.use_spectral_tok)}",
        f"xattn={int(cfg.use_cross_attn)}",
        f"norm={cfg.norm_type.value}",
        f"post1x1={int(cfg.use_post_1x1)}",
        f"dim={cfg.dim}",
        f"L={args.latent}",
        f"hist={args.history}",
        f"lr={args.lr}",
        f"vw={args.loss_vrmse_w}",
        f"sw={args.loss_spec_w}",
        f"l1w={args.loss_l1_w}",
    ]
    return "_".join(bits)

def main():
    ddp_setup()
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--base', type=str, required=True)
        parser.add_argument('--datasets', type=str, default='', help='Comma-separated list for multi-dataset')
        parser.add_argument('--dataset', type=str, default='', help='Single dataset name')
        parser.add_argument('--split', type=str, default='train')
        parser.add_argument('--epochs', type=int, default=5)
        parser.add_argument('--steps_per_epoch', type=int, default=1000)
        parser.add_argument('--batch', type=int, default=2)
        parser.add_argument('--workers', type=int, default=4)
        parser.add_argument('--history', type=int, default=1)
        parser.add_argument('--latent', type=int, default=8)
        parser.add_argument('--dim', type=int, default=256)
        parser.add_argument('--lr', type=float, default=2e-4)
        parser.add_argument('--save_dir', type=str, default='./checkpoints')
        parser.add_argument('--eval_only', action='store_true')
        parser.add_argument('--checkpoint', type=str, default='')

        # ablations
        parser.add_argument('--abl_use_film', type=int, default=1)
        parser.add_argument('--abl_use_spectral_tok', type=int, default=1)
        parser.add_argument('--abl_use_cross_attn', type=int, default=1)
        parser.add_argument('--abl_backbone', type=str, default='mamba', choices=['mamba','transformer'])
        parser.add_argument('--abl_decoder', type=str, default='fno', choices=['fno','conv'])
        parser.add_argument('--abl_use_post_1x1', type=int, default=1)
        parser.add_argument('--abl_norm', type=str, default='layer', choices=['layer','none'])

        # loss weights
        parser.add_argument('--loss_vrmse_w', type=float, default=1.0)
        parser.add_argument('--loss_spec_w', type=float, default=0.2)
        parser.add_argument('--loss_l1_w', type=float, default=0.0)

        # debug / amp
        parser.add_argument('--no_amp', action='store_true')
        parser.add_argument('--log_every', type=int, default=200)

        args = parser.parse_args()

        seed_everything(42)
        os.makedirs(args.save_dir, exist_ok=True)

        cfg_over = dict(
            use_film=bool(args.abl_use_film),
            use_spectral_tok=bool(args.abl_use_spectral_tok),
            use_cross_attn=bool(args.abl_use_cross_attn),
            backbone_type={'mamba':Backbone.mamba,'transformer':Backbone.transformer}[args.abl_backbone],
            decoder_type={'fno':Decoder.fno,'conv':Decoder.conv}[args.abl_decoder],
            use_post_1x1=bool(args.abl_use_post_1x1),
            norm_type={'layer':NormType.layer,'none':NormType.none}[args.abl_norm],
        )

        if args.datasets:
            names = [x.strip() for x in args.datasets.split(',') if x.strip()]
        else:
            names = [args.dataset or 'euler_multi_quadrants']

        trainer = Trainer(
            well_base_path=args.base,
            dataset_names=names,
            history=args.history,
            latent_channels=args.latent,
            phoenix_dim=args.dim,
            batch_size=args.batch,
            num_workers=args.workers,
            lr=args.lr,
            cfg_overrides=cfg_over,
            loss_vrmse_w=args.loss_vrmse_w,
            loss_spec_w=args.loss_spec_w,
            loss_l1_w=args.loss_l1_w,
            no_amp=args.no_amp,
            log_every=args.log_every,
        )

        exp_name = exp_run_id(args, trainer.cfg)

        if args.checkpoint:
            sd = torch.load(args.checkpoint, map_location='cpu')
            trainer._unwrap(trainer.model).load_state_dict(sd, strict=False)
            if is_main_process():
                log.info(f"Loaded checkpoint: {args.checkpoint}")

        if args.eval_only:
            trainer._unwrap(trainer.model).eval(); trainer._unwrap(trainer.adapters).eval()
            rep = {}
            with torch.no_grad():
                for name in trainer.val_loaders.keys():
                    rep[name] = trainer.evaluate_loader(name, trainer.val_loaders[name], max_batches=None)
            if is_main_process():
                print(json.dumps(rep, indent=2))
            return

        for epoch in range(1, args.epochs + 1):
            if dist.is_initialized():
                for s in trainer.train_samplers.values():
                    if s is not None: s.set_epoch(epoch)
            trainer.train_one_epoch(args.steps_per_epoch)
            report = trainer.validate_all()
            if is_main_process():
                with open(os.path.join(trainer.val_report_dir, f"epoch_{epoch:03d}.json"), "w") as f:
                    json.dump(report, f, indent=2)
            trainer.save_checkpoint(args.save_dir, epoch)

        if is_main_process():
            log.info("Done foundation pretraining across datasets.")
    finally:
        ddp_cleanup()

if __name__ == "__main__":
    main()
