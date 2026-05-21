"""
Run detection on a full-resolution image.

Strategy:
  1. Slice image into overlapping tiles (same grid as training).
  2. Run model on each tile.
  3. Shift predicted boxes back to image coordinates.
  4. Merge all tiles with per-class NMS to suppress duplicates from overlap zones.

Usage:
    python inference.py DATA/FH020_step0_0_0002.png
    python inference.py DATA/FH020_step0_0_0002.png --checkpoint checkpoints/best.pt
    python inference.py DATA/FH020_step0_0_0002.png --output detections.json
"""

import argparse
import json
from pathlib import Path

import torch
from torchvision.ops import nms
import torchvision.transforms.functional as TF
from PIL import Image

import config
from data_prep import tile_positions, open_as_uint8
from model import build_model

IDX_TO_LABEL = {i + 1: l for i, l in enumerate(config.LABELS)}


def load_model(checkpoint_path: str, device: torch.device):
    model = build_model(pretrained=False).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model


def detect(image_path: str, checkpoint_path: str,
           score_threshold: float = None, nms_iou: float = None):
    score_threshold = score_threshold if score_threshold is not None else config.SCORE_THRESHOLD
    nms_iou         = nms_iou         if nms_iou         is not None else config.NMS_IOU_THRESHOLD

    device = torch.device(config.DEVICE)
    model  = load_model(checkpoint_path, device)

    img_gray = open_as_uint8(image_path)
    W, H = img_gray.size

    xs = tile_positions(W, config.TILE_SIZE, config.STRIDE)
    ys = tile_positions(H, config.TILE_SIZE, config.STRIDE)

    all_boxes:  list[torch.Tensor] = []
    all_scores: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    with torch.no_grad():
        for ty in ys:
            for tx in xs:
                crop = img_gray.crop((tx, ty, tx + config.TILE_SIZE, ty + config.TILE_SIZE))
                rgb  = Image.merge('RGB', (crop, crop, crop))
                tensor = TF.to_tensor(rgb).unsqueeze(0).to(device)

                preds = model(tensor)[0]

                keep = preds['scores'] >= score_threshold
                if not keep.any():
                    continue

                boxes  = preds['boxes'][keep].clone()
                scores = preds['scores'][keep]
                labels = preds['labels'][keep]

                # Shift to full-image coordinates
                boxes[:, [0, 2]] += tx
                boxes[:, [1, 3]] += ty

                all_boxes.append(boxes.cpu())
                all_scores.append(scores.cpu())
                all_labels.append(labels.cpu())

    if not all_boxes:
        return {'boxes': [], 'scores': [], 'labels': [], 'label_names': []}

    all_boxes_t  = torch.cat(all_boxes)
    all_scores_t = torch.cat(all_scores)
    all_labels_t = torch.cat(all_labels)

    # Per-class NMS
    final_boxes, final_scores, final_labels = [], [], []
    for cls in all_labels_t.unique():
        mask = all_labels_t == cls
        keep = nms(all_boxes_t[mask], all_scores_t[mask], nms_iou)
        final_boxes.append(all_boxes_t[mask][keep])
        final_scores.append(all_scores_t[mask][keep])
        final_labels.append(all_labels_t[mask][keep])

    final_boxes_t  = torch.cat(final_boxes)
    final_scores_t = torch.cat(final_scores)
    final_labels_t = torch.cat(final_labels)

    return {
        'boxes':       final_boxes_t.numpy().tolist(),
        'scores':      final_scores_t.numpy().tolist(),
        'labels':      final_labels_t.numpy().tolist(),
        'label_names': [IDX_TO_LABEL[l] for l in final_labels_t.numpy().tolist()],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('image',             help='Path to input PNG image')
    parser.add_argument('--checkpoint',      default=str(config.CHECKPOINTS_DIR / 'best.pt'))
    parser.add_argument('--score_threshold', type=float, default=config.SCORE_THRESHOLD)
    parser.add_argument('--nms_iou',         type=float, default=config.NMS_IOU_THRESHOLD)
    parser.add_argument('--output',          default=None, help='Save detections to JSON file')
    args = parser.parse_args()

    result = detect(args.image, args.checkpoint, args.score_threshold, args.nms_iou)

    n = len(result['boxes'])
    print(f"Detected {n} objects in {args.image}")
    for box, score, name in zip(result['boxes'], result['scores'], result['label_names']):
        print(f"  {name:20s} score={score:.3f}  "
              f"box=[{box[0]:.0f},{box[1]:.0f},{box[2]:.0f},{box[3]:.0f}]")

    if args.output:
        detections = [
            {'label': name, 'score': score, 'box': box}
            for name, score, box in zip(
                result['label_names'], result['scores'], result['boxes']
            )
        ]
        with open(args.output, 'w') as f:
            json.dump(detections, f, indent=2)
        print(f"\nSaved {n} detections to {args.output}")


if __name__ == '__main__':
    main()
