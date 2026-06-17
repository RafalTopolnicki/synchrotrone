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
from tqdm import tqdm

import config
from dataset import TileDataset
from model import build_model, BACKBONES
from viz import log_sample_images, save_prediction_tiles
from evaluate import evaluate_all_splits, compute_per_image_stats


# ── data ─────────────────────────────────────────────────────────────────────

def collate_fn(batch):
    return tuple(zip(*batch))


def make_loaders(batch_size: int, merge_dunes: bool = False, rot180: bool = False):
    train_ds = TileDataset('train', augment=True,  merge_dunes=merge_dunes, rot180=rot180)
    val_ds   = TileDataset('val',   augment=False, merge_dunes=merge_dunes)
    test_ds  = TileDataset('test',  augment=False, merge_dunes=merge_dunes)

    kw = dict(num_workers=config.NUM_WORKERS, collate_fn=collate_fn)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  **kw)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, **kw)
    return train_loader, val_loader, train_ds, val_ds, test_ds


# ── training step ─────────────────────────────────────────────────────────────

def run_epoch(model, loader, optimizer, device, desc: str = ''):
    """One pass; returns per-loss averages. optimizer=None → val pass (no grad)."""
    is_train = optimizer is not None
    model.train()   # Faster R-CNN only returns losses in train mode
    totals: dict[str, float] = {}
    step = 0

    bar = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
    with torch.set_grad_enabled(is_train):
        for images, targets in bar:
            images  = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            step += 1
            for k, v in loss_dict.items():
                totals[k] = totals.get(k, 0.0) + v.item()

            # Running averages shown in the progress bar
            bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'cls':  f"{loss_dict.get('loss_classifier', torch.tensor(0)).item():.3f}",
                'box':  f"{loss_dict.get('loss_box_reg',    torch.tensor(0)).item():.3f}",
                'rpn':  f"{loss_dict.get('loss_objectness', torch.tensor(0)).item():.3f}",
            })

    n = max(step, 1)
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
    parser.add_argument('--backbone',    default='resnet50v2',
                        choices=list(BACKBONES),
                        help='Feature extractor backbone')
    parser.add_argument('--epochs',      type=int,   default=config.NUM_EPOCHS,
                        help='Maximum number of training epochs')
    parser.add_argument('--lr',          type=float, default=config.LR)
    parser.add_argument('--batch_size',  type=int,   default=config.BATCH_SIZE)
    parser.add_argument('--patience',    type=int,   default=7,
                        help='Early stopping patience (0 = disabled)')
    parser.add_argument('--merge_dunes', action='store_true', default=True,
                        help='Merge CoR_dune_up and CoR_dune_down into a single CoR_dune class (default: on)')
    parser.add_argument('--no_merge_dunes', action='store_false', dest='merge_dunes',
                        help='Disable dune class merging')
    parser.add_argument('--rot180', action='store_true', default=True,
                        help='Add random 180° rotation to training augmentation (default: on)')
    parser.add_argument('--no_rot180', action='store_false', dest='rot180',
                        help='Disable 180° rotation augmentation')
    parser.add_argument('--weight_decay', type=float, default=config.LR_WEIGHT_DECAY,
                        help='Optimizer weight decay (default: config.LR_WEIGHT_DECAY)')
    parser.add_argument('--name', default='',
                        help='Descriptive suffix appended to the run directory name')
    parser.add_argument('--freeze_epochs', type=int, default=10,
                        help='Freeze backbone for this many epochs then unfreeze (0 = train all layers)')
    parser.add_argument('--save_checkpoints', action='store_true',
                        help='Save best.pt and last.pt to disk (default: off)')
    parser.add_argument('--resume',      default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--log_img_every', type=int, default=5,
                        help='Log sample images to TensorBoard every N epochs')
    parser.add_argument('--score_threshold', type=float, default=0.05,
                        help='Minimum score for a box to be kept (box_score_thresh in Faster R-CNN)')
    parser.add_argument('--eval_only', action='store_true',
                        help='Skip training; load --resume checkpoint and run evaluation only')
    args = parser.parse_args()

    # ── run directory ──────────────────────────────────────────────────────
    run_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.backbone}"
    if args.name:
        run_name = f"{run_name}_{args.name}"
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
    train_loader, val_loader, train_ds, val_ds, test_ds = make_loaders(args.batch_size, args.merge_dunes, args.rot180)
    print(f"Tiles — train: {len(train_ds)}  val: {len(val_ds)}  test: {len(test_ds)}")

    # ── model / optimiser ──────────────────────────────────────────────────
    num_classes = 2 if args.merge_dunes else config.NUM_CLASSES
    model = build_model(backbone=args.backbone, pretrained=True, num_classes=num_classes,
                        box_score_thresh=args.score_threshold).to(device)

    def _make_optimizer(lr):
        params = [p for p in model.parameters() if p.requires_grad]
        return torch.optim.AdamW(params, lr=lr, weight_decay=args.weight_decay)

    def _make_scheduler(opt):
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode='min', factor=0.5, patience=5, min_lr=1e-6)

    if args.freeze_epochs > 0:
        for name, p in model.named_parameters():
            if 'backbone' in name:
                p.requires_grad = False
        print(f"Backbone frozen for first {args.freeze_epochs} epochs")

    optimizer = _make_optimizer(args.lr)
    scheduler = _make_scheduler(optimizer)
    stopper   = EarlyStopping(args.patience)

    start_epoch      = 1
    best_val         = float('inf')
    best_model_state = None
    history          = []

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
    if args.eval_only:
        print("Eval-only mode — skipping training, using loaded checkpoint weights")

    for epoch in range(start_epoch if not args.eval_only else args.epochs + 1, args.epochs + 1):
        t0 = time.time()

        # Unfreeze backbone after freeze_epochs
        if args.freeze_epochs > 0 and epoch == args.freeze_epochs + 1:
            for p in model.parameters():
                p.requires_grad = True
            optimizer = _make_optimizer(args.lr)
            scheduler = _make_scheduler(optimizer)
            print("Backbone unfrozen — all layers now training")

        print(f"\n{'─'*70}")
        print(f"Epoch {epoch:03d}/{args.epochs}  "
              f"backbone={args.backbone}  bs={args.batch_size}  lr={optimizer.param_groups[0]['lr']:.2e}")

        train_losses = run_epoch(model, train_loader, optimizer, device,
                                 desc=f"  train {epoch:03d}")
        val_losses   = run_epoch(model, val_loader,   None,      device,
                                 desc=f"  val   {epoch:03d}")

        train_total = sum(train_losses.values())
        val_total   = sum(val_losses.values())
        elapsed     = time.time() - t0

        scheduler.step(val_total)

        # ── console summary ───────────────────────────────────────────────
        print(f"  {'loss':35s}  {'train':>10}  {'val':>10}")
        print(f"  {'─'*57}")
        print(f"  {'total':35s}  {train_total:10.4f}  {val_total:10.4f}")
        for k in train_losses:
            print(f"  {k:35s}  {train_losses[k]:10.4f}  {val_losses.get(k, 0):10.4f}")
        print(f"  elapsed: {elapsed:.0f}s")

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
                              val_ds, device, epoch, n=4,
                              merge_dunes=args.merge_dunes)

        # ── checkpoints ───────────────────────────────────────────────────
        if val_total < best_val:
            best_val         = val_total
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            if args.save_checkpoints:
                ckpt = dict(epoch=epoch, model=model.state_dict(),
                            optimizer=optimizer.state_dict(),
                            val_loss=val_total, best_val=best_val,
                            backbone=_backbone)
                torch.save(ckpt, ckpt_dir / 'best.pt')
            print(f"  *** new best val={best_val:.4f}"
                  + (" — saved best.pt" if args.save_checkpoints else ""))

        if args.save_checkpoints:
            ckpt = dict(epoch=epoch, model=model.state_dict(),
                        optimizer=optimizer.state_dict(),
                        val_loss=val_total, best_val=best_val,
                        backbone=_backbone)
            torch.save(ckpt, ckpt_dir / 'last.pt')

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
    if not args.eval_only:
        print("\nLoading best weights for evaluation…")
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        elif args.save_checkpoints:
            best_ckpt = torch.load(ckpt_dir / 'best.pt', map_location=device)
            model.load_state_dict(best_ckpt['model'])

    # ── visualise predictions ──────────────────────────────────────────────
    print("Generating prediction visualisations…")
    for split, ds in [('val', val_ds), ('test', test_ds)]:
        save_prediction_tiles(model, ds, device, pred_dir / split, n=10,
                              merge_dunes=args.merge_dunes)

    # ── metrics on all splits ─────────────────────────────────────────────
    print("\nComputing metrics…")
    evaluate_all_splits(model, device, run_dir, merge_dunes=args.merge_dunes)

    # ── per-image GT vs predicted counts ──────────────────────────────────
    print("\nComputing per-image stats…")
    compute_per_image_stats(model, run_dir, merge_dunes=args.merge_dunes)

    print(f"\nDone. Best val loss: {best_val:.4f}")
    print(f"Run saved to: {run_dir}")


if __name__ == '__main__':
    main()
