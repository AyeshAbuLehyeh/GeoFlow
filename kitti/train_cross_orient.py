from __future__ import annotations

import argparse
import os
import time

import torch

try:
    from .data import load_train_data
    from .engine import create_optimizer, save_checkpoint, set_seed, train_one_epoch, write_json
    from .models import GeoFlowKITTIOrientation
except ImportError:  # pragma: no cover - standalone execution fallback
    from data import load_train_data
    from engine import create_optimizer, save_checkpoint, set_seed, train_one_epoch, write_json
    from models import GeoFlowKITTIOrientation


def parse_args():
    parser = argparse.ArgumentParser(description="Train GeoFlow on KITTI with orientation prediction")
    parser.add_argument("--train-list", default="./train_files.txt", type=str)
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--backbone-lr", default=1e-4, type=float)
    parser.add_argument("--weight-decay", default=1e-2, type=float)
    parser.add_argument("--max-grad-norm", default=1.0, type=float)
    parser.add_argument("--lambda-r", default=1.0, type=float)
    parser.add_argument("--backbone", default="efficientnet_b0", type=str)
    parser.add_argument("--d-model", default=128, type=int)
    parser.add_argument("--sat-size", default=512, type=int)
    parser.add_argument("--rotation-range", default=10.0, type=float)
    parser.add_argument("--checkpoint-dir", default="checkpoints", type=str)
    parser.add_argument("--log-dir", default="logs", type=str)
    parser.add_argument("--save-every", default=5, type=int)
    parser.add_argument("--seed", default=42, type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_name = f"kitti_orient_{args.backbone.replace('_', '')}_d{args.d_model}_{timestamp}"
    run_dir = os.path.join(args.checkpoint_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    train_loader = load_train_data(file_path=args.train_list, batch_size=args.batch_size, use_augmentation=True)
    print(f"Training on {len(train_loader.dataset)} samples on {device}.")

    model = GeoFlowKITTIOrientation(d_model=args.d_model, backbone=args.backbone, sat_size=args.sat_size).to(device)
    optimizer = create_optimizer(model, lr=args.lr, backbone_lr=args.backbone_lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    write_json(os.path.join(run_dir, "config.json"), vars(args))
    log_path = os.path.join(run_dir, "train_log.txt")
    best_train_loss = float("inf")

    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write("GeoFlow KITTI orientation-aware training run\n")
        log_file.write(f"Device: {device}\n")
        log_file.write(f"Backbone: {args.backbone}\n")
        log_file.write(f"Batch size: {args.batch_size}\n")
        log_file.write(f"Learning rates: main={args.lr}, backbone={args.backbone_lr}\n")
        log_file.write(f"Weight decay: {args.weight_decay}\n")
        log_file.write(f"Lambda_R: {args.lambda_r}\n")
        log_file.write(f"Rotation range: {args.rotation_range}\n\n")

        for epoch_idx in range(args.epochs):
            print(f"\n----- Epoch {epoch_idx + 1}/{args.epochs} -----")
            metrics = train_one_epoch(
                model=model,
                dataloader=train_loader,
                optimizer=optimizer,
                device=device,
                lambda_r=args.lambda_r,
                max_grad_norm=args.max_grad_norm,
                rotation_range=args.rotation_range,
            )
            scheduler.step()

            log_message = (
                f"Epoch {epoch_idx + 1:03d} | total={metrics['total']:.6f} | theta={metrics['theta']:.6f} | "
                f"r={metrics['r']:.6f} | orient={metrics['orientation']:.6f} | "
                f"lr_main={optimizer.param_groups[1]['lr']:.3e} | lr_backbone={optimizer.param_groups[0]['lr']:.3e}"
            )
            print(log_message)
            log_file.write(log_message + "\n")
            log_file.flush()

            save_checkpoint(os.path.join(run_dir, "last.pth"), model, optimizer=optimizer, scheduler=scheduler, epoch=epoch_idx + 1, best_metric=best_train_loss, args=args)

            if metrics["total"] < best_train_loss:
                best_train_loss = metrics["total"]
                best_path = os.path.join(run_dir, "best.pth")
                save_checkpoint(best_path, model, optimizer=optimizer, scheduler=scheduler, epoch=epoch_idx + 1, best_metric=best_train_loss, args=args)
                print(f"-> New best checkpoint saved: {best_path}")

            if args.save_every > 0 and (epoch_idx + 1) % args.save_every == 0:
                periodic_path = os.path.join(run_dir, f"epoch_{epoch_idx + 1:03d}.pth")
                save_checkpoint(periodic_path, model, optimizer=optimizer, scheduler=scheduler, epoch=epoch_idx + 1, best_metric=best_train_loss, args=args)
                print(f"-> Saved periodic checkpoint: {periodic_path}")

    print(f"Training complete. Outputs saved to {run_dir}.")


if __name__ == "__main__":
    main()