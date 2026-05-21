"""
Train Faster R-CNN on tiled synchrotron images.

Prerequisites:
    python data_prep.py

Usage:
    python train.py
    python train.py --backbone resnet50v2 --epochs 50 --lr 0.001 --batch_size 8
    python train.py --resume runs/20260521_120000/checkpoints/last.pt

Backbone choices: resnet50 (default), resnet50v2, mobilenet

Each run is saved to:
    runs/<YYYYMMDD_HHMMSS>/
        args.json
        tensorboard/          ← tensorboard --logdir runs
        checkpoints/
            best.pt  last.pt  history.json
        predictions/
            val/   pred_00.png … pred_09.png
            test/  pred_00.png … pred_09.png
        metrics.json          ← mAP / AP / P / R per split × class
        per_image_stats.json  ← GT vs predicted counts per source image
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import config
from dataset import TileDataset
from model import build_model, BACKBONES
from viz import log_sample_images, save_prediction_tiles
from evaluate import evaluate_all_splits, compute_per_image_stats


# ── data ─────────────────────────────────────────────────────────────────────

def collate_fn(batch):
    return tuple(zip(*batch))


def make_loaders(batch_size: int):
    train_ds = TileDataset('train', augment=True)
    val_ds   = TileDataset('val',   augment=False)
    test_ds  = TileDataset('test',  augment=False)

    kw = dict(num_workers=config.NUM_WORKERS, collate_fn=collate_fn)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  **kw)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, **kw)
    return train_loader, val_loader, train_ds, val_ds, test_ds


# ── training step ─────────────────────────────────────────────────────────────

def run_epoch(model, loader, optimizer, device):
    """One pass; returns per-loss averages. optimizer=None → val pass (no grad)."""
    model.train()   # Faster R-CNN only returns losses in train mode
    totals: dict[str, float] = {}
    with torch.set_grad_enabled(optimizer is not None):
        for images, targets in loader:
            images  = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())

            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            for k, v in loss_dict.items():
                totals[k] = totals.get(k, 0.0) + v.item()

    n = max(len(loader), 1)
    return {k: v / n for k, v in totals.items()}


# ── early stopping ────────────────────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience: int):
        self.patience = patience
        self.best     = float('inf')
        self.counter  = 0

    def step(self, val_loss: float) -> bool:
        """Returns True when training should stop."""
        if self.patience <= 0:
            return False
        if val_loss < self.best:
            self.best    = val_loss
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone',    default='resnet50',
                        choices=list(BACKBONES),
                        help='Feature extractor backbone')
    parser.add_argument('--epochs',      type=int,   default=config.NUM_EPOCHS,
                        help='Maximum number of training epochs')
    parser.add_argument('--lr',          type=float, default=config.LR)
    parser.add_argument('--batch_size',  type=int,   default=config.BATCH_SIZE)
    parser.add_argument('--patience',    type=int,   default=7,
                        help='Early stopping patience (0 = disabled)')
    parser.add_argument('--resume',      default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--log_img_every', type=int, default=5,
                        help='Log sample images to TensorBoard every N epochs')
    args = parser.parse_args()

    # ── run directory ──────────────────────────────────────────────────────
    run_name = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir  = Path('runs') / run_name
    ckpt_dir = run_dir / 'checkpoints'
    pred_dir = run_dir / 'predictions'
    tb_dir   = run_dir / 'tensorboard'
    for d in (ckpt_dir, pred_dir, tb_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Save the exact args used for this run
    (run_dir / 'args.json').write_text(json.dumps(vars(args), indent=2))

    print(f"Run directory : {run_dir}")
    print(f"Backbone      : {args.backbone}")
    print(f"TensorBoard   : tensorboard --logdir runs")

    device = torch.device(config.DEVICE)
    print(f"Device        : {device}")

    # ── data ───────────────────────────────────────────────────────────────
    train_loader, val_loader, train_ds, val_ds, test_ds = make_loaders(args.batch_size)
    print(f"Tiles — train: {len(train_ds)}  val: {len(val_ds)}  test: {len(test_ds)}")

    # ── model / optimiser ──────────────────────────────────────────────────
    model = build_model(backbone=args.backbone, pretrained=True).to(device)

    params    = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=args.lr,
        momentum=config.LR_MOMENTUM,
        weight_decay=config.LR_WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config.LR_STEP_SIZE, gamma=config.LR_GAMMA,
    )
    stopper   = EarlyStopping(args.patience)

    start_epoch = 1
    best_val    = float('inf')
    history     = []

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
        best_val    = ckpt.get('best_val', float('inf'))
        print(f"Resumed from epoch {ckpt['epoch']}")

    # Store backbone name in checkpoint for reference
    _backbone = args.backbone

    writer = SummaryWriter(log_dir=str(tb_dir))

    # ── training loop ──────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        train_losses = run_epoch(model, train_loader, optimizer, device)
        val_losses   = run_epoch(model, val_loader,   None,      device)
        scheduler.step()

        train_total = sum(train_losses.values())
        val_total   = sum(val_losses.values())
        elapsed     = time.time() - t0

        # ── console ──────────────────────────────────────────────────────
        print(f"\nEpoch {epoch:03d}/{args.epochs}  "
              f"train={train_total:.4f}  val={val_total:.4f}  ({elapsed:.0f}s)")
        for k in train_losses:
            print(f"  {k:35s}  train={train_losses[k]:.4f}  "
                  f"val={val_losses.get(k, 0):.4f}")

        # ── TensorBoard scalars ───────────────────────────────────────────
        writer.add_scalars('loss/total',
                           {'train': train_total, 'val': val_total}, epoch)
        for k in train_losses:
            writer.add_scalars(f'loss/{k}',
                               {'train': train_losses[k],
                                'val':   val_losses.get(k, 0)}, epoch)
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        # ── TensorBoard images ────────────────────────────────────────────
        if epoch % args.log_img_every == 0 or epoch == 1:
            log_sample_images(writer, 'val/predictions', model,
                              val_ds, device, epoch, n=4)

        # ── checkpoints ───────────────────────────────────────────────────
        ckpt = dict(epoch=epoch, model=model.state_dict(),
                    optimizer=optimizer.state_dict(),
                    val_loss=val_total, best_val=best_val,
                    backbone=_backbone)
        torch.save(ckpt, ckpt_dir / 'last.pt')

        if val_total < best_val:
            best_val    = val_total
            ckpt['best_val'] = best_val
            torch.save(ckpt, ckpt_dir / 'best.pt')
            print(f"  *** new best val={best_val:.4f} — saved best.pt")

        row = {'epoch': epoch, 'train': train_losses, 'val': val_losses}
        history.append(row)
        (ckpt_dir / 'history.json').write_text(json.dumps(history, indent=2))

        # ── early stopping ────────────────────────────────────────────────
        if stopper.step(val_total):
            print(f"\nEarly stopping triggered (patience={args.patience}, "
                  f"no improvement for {args.patience} epochs)")
            break

    writer.close()

    # ── post-training: load best weights ──────────────────────────────────
    print("\nLoading best checkpoint for evaluation…")
    best_ckpt = torch.load(ckpt_dir / 'best.pt', map_location=device)
    model.load_state_dict(best_ckpt['model'])

    # ── visualise predictions ──────────────────────────────────────────────
    print("Generating prediction visualisations…")
    for split, ds in [('val', val_ds), ('test', test_ds)]:
        save_prediction_tiles(model, ds, device, pred_dir / split, n=10)

    # ── metrics on all splits ─────────────────────────────────────────────
    print("\nComputing metrics…")
    evaluate_all_splits(model, device, run_dir)

    # ── per-image GT vs predicted counts ──────────────────────────────────
    print("\nComputing per-image stats…")
    compute_per_image_stats(model, run_dir)

    print(f"\nDone. Best val loss: {best_val:.4f}")
    print(f"Run saved to: {run_dir}")


if __name__ == '__main__':
    main()
