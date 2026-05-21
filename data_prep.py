"""
Tiles source images, clips annotations to each tile, and writes a split index.

Splits are done at the base-image level: FH020_step0_0_0002 and
FH020_step0_0_0002_sat are considered the same base image and always land in
the same split, preventing data leakage through augmented variants.

Run once before training:
    python data_prep.py
"""

import json
import random
from pathlib import Path
import numpy as np
from PIL import Image

import config


def open_as_uint8(path) -> Image.Image:
    """Load any-bit-depth grayscale PNG as an 8-bit ('L') PIL image.

    16-bit images (uint16) are converted by taking the high byte (>> 8),
    which preserves the full tonal range as 0-255.
    """
    arr = np.array(Image.open(path))
    if arr.dtype == np.uint16:
        arr = (arr >> 8).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = ((arr.astype(np.float32) - arr.min()) /
               max(arr.max() - arr.min(), 1) * 255).astype(np.uint8)
    return Image.fromarray(arr, mode='L')


def get_base_name(filename: str) -> str:
    stem = Path(filename).stem
    return stem[:-4] if stem.endswith('_sat') else stem


def labelme_rect_to_xyxy(points):
    """Two-point labelme rectangle → (x1, y1, x2, y2)."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def tile_positions(img_size: int, tile_size: int, stride: int) -> list[int]:
    positions = list(range(0, img_size - tile_size, stride))
    if not positions or positions[-1] + tile_size < img_size:
        positions.append(max(0, img_size - tile_size))
    return positions


def clip_box_to_tile(box, tx, ty, tile_size):
    """
    Clip box (x1,y1,x2,y2) to tile [tx, tx+tile_size) x [ty, ty+tile_size).
    Returns tile-local coordinates, or None if overlap < MIN_BBOX_OVERLAP.
    """
    x1, y1, x2, y2 = box
    cx1 = max(x1, tx)
    cy1 = max(y1, ty)
    cx2 = min(x2, tx + tile_size)
    cy2 = min(y2, ty + tile_size)
    if cx2 <= cx1 or cy2 <= cy1:
        return None
    orig_area = (x2 - x1) * (y2 - y1)
    if orig_area > 0 and (cx2 - cx1) * (cy2 - cy1) / orig_area < config.MIN_BBOX_OVERLAP:
        return None
    return cx1 - tx, cy1 - ty, cx2 - tx, cy2 - ty


def main():
    config.TILES_DIR.mkdir(parents=True, exist_ok=True)
    config.CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    ann_files = sorted(config.ANNOTATIONS_DIR.glob('*.json'))
    if not ann_files:
        raise FileNotFoundError(f"No annotation JSONs found in {config.ANNOTATIONS_DIR}")

    # Group annotation files by base image name
    groups: dict[str, list[Path]] = {}
    for af in ann_files:
        groups.setdefault(get_base_name(af.name), []).append(af)

    base_names = sorted(groups)
    random.seed(config.RANDOM_SEED)
    random.shuffle(base_names)

    n = len(base_names)
    n_train = max(1, round(n * config.TRAIN_RATIO))
    n_val = max(1, round(n * config.VAL_RATIO))

    split_map = (
        {b: 'train' for b in base_names[:n_train]}
        | {b: 'val'   for b in base_names[n_train:n_train + n_val]}
        | {b: 'test'  for b in base_names[n_train + n_val:]}
    )

    train_n = sum(1 for s in split_map.values() if s == 'train')
    val_n   = sum(1 for s in split_map.values() if s == 'val')
    test_n  = sum(1 for s in split_map.values() if s == 'test')
    print(f"Base-image split: {train_n} train / {val_n} val / {test_n} test")
    print(f"  train: {[b for b,s in split_map.items() if s=='train']}")
    print(f"  val:   {[b for b,s in split_map.items() if s=='val']}")
    print(f"  test:  {[b for b,s in split_map.items() if s=='test']}")

    label_to_idx = {l: i + 1 for i, l in enumerate(config.LABELS)}
    tile_index: dict[str, list] = {'train': [], 'val': [], 'test': []}

    for base, file_list in groups.items():
        split = split_map[base]

        for ann_file in sorted(file_list):
            stem = ann_file.stem
            img_path = config.IMAGES_DIR / f'{stem}.png'
            if not img_path.exists():
                print(f"  WARNING: image not found: {img_path}, skipping")
                continue

            with open(ann_file) as f:
                ann = json.load(f)

            boxes_raw = []
            for shape in ann['shapes']:
                lbl = shape['label']
                if lbl not in label_to_idx:
                    continue
                boxes_raw.append((label_to_idx[lbl], labelme_rect_to_xyxy(shape['points'])))

            img = open_as_uint8(img_path)
            W, H = img.size
            xs = tile_positions(W, config.TILE_SIZE, config.STRIDE)
            ys = tile_positions(H, config.TILE_SIZE, config.STRIDE)

            n_tiles = 0
            for ty in ys:
                for tx in xs:
                    tile_img = img.crop((tx, ty, tx + config.TILE_SIZE, ty + config.TILE_SIZE))

                    tile_boxes, tile_labels = [], []
                    for cls_idx, box in boxes_raw:
                        clipped = clip_box_to_tile(box, tx, ty, config.TILE_SIZE)
                        if clipped is not None:
                            tile_boxes.append(list(clipped))
                            tile_labels.append(cls_idx)

                    tile_name = f'{stem}_tx{tx}_ty{ty}.png'
                    tile_img.save(config.TILES_DIR / tile_name)

                    tile_index[split].append({
                        'tile_path': str(config.TILES_DIR / tile_name),
                        'source_image': stem,
                        'tx': tx,
                        'ty': ty,
                        'boxes': tile_boxes,
                        'labels': tile_labels,
                    })
                    n_tiles += 1

            print(f"  [{split}] {stem}: {n_tiles} tiles, {len(boxes_raw)} annotations")

    with open(config.SPLITS_FILE, 'w') as f:
        json.dump(tile_index, f)

    print()
    for split, tiles in tile_index.items():
        n_ann = sum(1 for t in tiles if t['labels'])
        print(f"{split:5s}: {len(tiles):4d} tiles  ({n_ann} with annotations)")
    print(f"\nSplit index saved to {config.SPLITS_FILE}")


if __name__ == '__main__':
    main()
