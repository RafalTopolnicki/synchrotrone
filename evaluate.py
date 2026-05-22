"""
Post-training evaluation utilities.

Two outputs produced after training:
  metrics.json         — mAP / AP / precision / recall per split × class
  per_image_stats.json — GT vs predicted counts per source image × class
"""

import json
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.ops import nms
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

import config
from data_prep import tile_positions, open_as_uint8

IDX_TO_LABEL = {i + 1: l for i, l in enumerate(config.LABELS)}
LABEL_TO_IDX = {l: i + 1 for i, l in enumerate(config.LABELS)}


# ── IoU / AP helpers ─────────────────────────────────────────────────────────

def _box_iou(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    """Pairwise IoU between [N,4] and [M,4] tensors (xyxy format)."""
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])

    inter_x1 = torch.max(boxes_a[:, None, 0], boxes_b[None, :, 0])
    inter_y1 = torch.max(boxes_a[:, None, 1], boxes_b[None, :, 1])
    inter_x2 = torch.min(boxes_a[:, None, 2], boxes_b[None, :, 2])
    inter_y2 = torch.min(boxes_a[:, None, 3], boxes_b[None, :, 3])

    inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / union.clamp(min=1e-6)


def _compute_ap(rec: list, prec: list) -> float:
    """Area under PR curve (all-points interpolation, PASCAL VOC style)."""
    rec  = [0.0] + rec  + [1.0]
    prec = [0.0] + prec + [0.0]
    for i in range(len(prec) - 2, -1, -1):
        prec[i] = max(prec[i], prec[i + 1])
    ap = sum((rec[i] - rec[i - 1]) * prec[i] for i in range(1, len(rec)))
    return float(ap)


# ── tile-level mAP ────────────────────────────────────────────────────────────

def compute_map(model, dataset, device,
                iou_threshold: float = 0.5,
                idx_to_label: dict = None) -> dict:
    """
    Compute per-class AP and mAP on a TileDataset.

    Returns:
        {
          'mAP': float,
          'per_class': {
              label: {'AP': float, 'precision': float, 'recall': float,
                      'n_gt': int, 'n_pred': int}
          }
        }
    """
    if idx_to_label is None:
        idx_to_label = IDX_TO_LABEL

    def collate(b): return tuple(zip(*b))

    loader = DataLoader(dataset, batch_size=4, shuffle=False,
                        num_workers=config.NUM_WORKERS, collate_fn=collate)

    # per_class[cls] = {'preds': [(score, is_tp), ...], 'n_gt': int, 'n_pred': int}
    per_class: dict[int, dict] = defaultdict(lambda: {'preds': [], 'n_gt': 0, 'n_pred': 0})

    model.eval()
    with torch.no_grad():
        for images, targets in tqdm(loader, desc='  evaluating', leave=False, dynamic_ncols=True):
            preds_list = model([img.to(device) for img in images])

            for target, preds in zip(targets, preds_list):
                gt_boxes  = target['boxes']
                gt_labels = target['labels']

                for lbl in gt_labels.tolist():
                    per_class[lbl]['n_gt'] += 1

                # Sort predictions by score descending (move to CPU to match GT)
                order       = preds['scores'].argsort(descending=True)
                pred_boxes  = preds['boxes'][order].cpu()
                pred_scores = preds['scores'][order].cpu()
                pred_labels = preds['labels'][order].cpu()

                gt_matched = torch.zeros(len(gt_labels), dtype=torch.bool)

                for pb, ps, pl in zip(pred_boxes, pred_scores, pred_labels):
                    pl_i = int(pl)
                    per_class[pl_i]['n_pred'] += 1

                    gt_mask = gt_labels == pl_i
                    is_tp   = False

                    if gt_mask.any():
                        gt_cls   = gt_boxes[gt_mask]
                        ious     = _box_iou(pb.unsqueeze(0), gt_cls)[0]
                        best_iou, best_j = ious.max(0)
                        if best_iou >= iou_threshold:
                            gt_idx = gt_mask.nonzero(as_tuple=True)[0][best_j]
                            if not gt_matched[gt_idx]:
                                gt_matched[gt_idx] = True
                                is_tp = True

                    per_class[pl_i]['preds'].append((float(ps), is_tp))

    result_per_class = {}
    aps = []

    for cls_idx, data in per_class.items():
        if data['n_gt'] == 0:
            continue
        sorted_preds = sorted(data['preds'], key=lambda x: -x[0])
        tp = fp = 0
        rec_list, prec_list = [], []
        for _, is_tp in sorted_preds:
            if is_tp: tp += 1
            else:      fp += 1
            rec_list.append(tp / data['n_gt'])
            prec_list.append(tp / (tp + fp))

        ap = _compute_ap(rec_list, prec_list)
        aps.append(ap)
        label = idx_to_label.get(cls_idx, str(cls_idx))
        result_per_class[label] = {
            'AP':        round(ap, 4),
            'precision': round(prec_list[-1] if prec_list else 0.0, 4),
            'recall':    round(rec_list[-1]  if rec_list  else 0.0, 4),
            'n_gt':      data['n_gt'],
            'n_pred':    data['n_pred'],
        }

    return {
        'mAP':       round(sum(aps) / max(len(aps), 1), 4),
        'per_class': result_per_class,
    }


# ── per-image stats (full-image inference + NMS) ─────────────────────────────

def compute_per_image_stats(model, run_dir: Path,
                             score_threshold: float = None,
                             merge_dunes: bool = False) -> list[dict]:
    """
    For every source image (across all splits):
      - GT counts from annotation file
      - Predicted counts via full-image tiling + per-class NMS

    Writes run_dir/per_image_stats.json and returns the list.
    """
    score_threshold = score_threshold or config.SCORE_THRESHOLD
    _raw_dune_labels = {'CoR_dune_down', 'CoR_dune_up'}
    labels_to_report = ['CoR_dune'] if merge_dunes else config.LABELS
    idx_to_label     = {1: 'CoR_dune'} if merge_dunes else IDX_TO_LABEL

    with open(config.SPLITS_FILE) as f:
        split_index = json.load(f)

    # source_image → split
    image_to_split: dict[str, str] = {}
    for split, tiles in split_index.items():
        for tile in tiles:
            image_to_split[tile['source_image']] = split

    device = next(model.parameters()).device
    model.eval()

    results = []

    for stem in tqdm(sorted(image_to_split), desc='  per-image inference', dynamic_ncols=True):
        split     = image_to_split[stem]
        img_path  = config.IMAGES_DIR    / f'{stem}.png'
        ann_path  = config.ANNOTATIONS_DIR / f'{stem}.json'

        if not img_path.exists() or not ann_path.exists():
            print(f"  WARNING: missing file for {stem}, skipping")
            continue

        # ── GT counts ────────────────────────────────────────────────────
        with open(ann_path) as f:
            ann = json.load(f)
        gt_counts: dict[str, int] = defaultdict(int)
        for shape in ann['shapes']:
            lbl = shape['label']
            if merge_dunes:
                if lbl in _raw_dune_labels:
                    gt_counts['CoR_dune'] += 1
            elif lbl in LABEL_TO_IDX:
                gt_counts[lbl] += 1

        # ── full-image inference ─────────────────────────────────────────
        img_gray = open_as_uint8(img_path)
        W, H = img_gray.size
        xs = tile_positions(W, config.TILE_SIZE, config.STRIDE)
        ys = tile_positions(H, config.TILE_SIZE, config.STRIDE)

        all_boxes:  list[torch.Tensor] = []
        all_scores: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []

        with torch.no_grad():
            for ty in ys:
                for tx in xs:
                    crop   = img_gray.crop((tx, ty,
                                            tx + config.TILE_SIZE,
                                            ty + config.TILE_SIZE))
                    rgb    = Image.merge('RGB', (crop, crop, crop))
                    tensor = TF.to_tensor(rgb).unsqueeze(0).to(device)

                    pred = model(tensor)[0]
                    keep = pred['scores'] >= score_threshold
                    if not keep.any():
                        continue

                    boxes = pred['boxes'][keep].clone().cpu()
                    boxes[:, [0, 2]] += tx
                    boxes[:, [1, 3]] += ty
                    all_boxes.append(boxes)
                    all_scores.append(pred['scores'][keep].cpu())
                    all_labels.append(pred['labels'][keep].cpu())

        pred_counts: dict[str, int] = defaultdict(int)
        if all_boxes:
            all_boxes_t  = torch.cat(all_boxes)
            all_scores_t = torch.cat(all_scores)
            all_labels_t = torch.cat(all_labels)

            for cls in all_labels_t.unique():
                mask  = all_labels_t == cls
                kept  = nms(all_boxes_t[mask], all_scores_t[mask],
                            config.NMS_IOU_THRESHOLD)
                lbl   = idx_to_label.get(int(cls), str(int(cls)))
                pred_counts[lbl] = int(len(kept))

        row = {
            'image': stem,
            'split': split,
            'gt':   {lbl: int(gt_counts.get(lbl, 0))   for lbl in labels_to_report},
            'pred': {lbl: int(pred_counts.get(lbl, 0))  for lbl in labels_to_report},
        }
        results.append(row)
        print(f"  [{split:5s}] {stem}")
        print(f"          gt  : { {k: v for k, v in row['gt'].items()  if v} }")
        print(f"          pred: { {k: v for k, v in row['pred'].items() if v} }")

    out_path = run_dir / 'per_image_stats.json'
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nPer-image stats saved → {out_path}")
    return results


# ── convenience: evaluate all splits ─────────────────────────────────────────

def evaluate_all_splits(model, device, run_dir: Path, merge_dunes: bool = False) -> dict:
    """
    Compute mAP on train / val / test and write run_dir/metrics.json.
    """
    from dataset import TileDataset

    idx_to_label = {1: 'CoR_dune'} if merge_dunes else IDX_TO_LABEL

    all_metrics = {}
    for split in ('train', 'val', 'test'):
        print(f"\nEvaluating {split} split …")
        ds = TileDataset(split, augment=False, merge_dunes=merge_dunes)
        all_metrics[split] = compute_map(model, ds, device, idx_to_label=idx_to_label)
        m = all_metrics[split]
        print(f"  mAP@0.5 = {m['mAP']:.4f}")
        for lbl, vals in m['per_class'].items():
            print(f"    {lbl:20s}  AP={vals['AP']:.4f}  "
                  f"P={vals['precision']:.4f}  R={vals['recall']:.4f}  "
                  f"GT={vals['n_gt']}  pred={vals['n_pred']}")

    out_path = run_dir / 'metrics.json'
    out_path.write_text(json.dumps(all_metrics, indent=2))
    print(f"\nMetrics saved → {out_path}")
    return all_metrics
