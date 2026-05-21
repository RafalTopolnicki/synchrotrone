"""
Visualization helpers shared by train.py and inference.py.
"""
import io

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import torch

import config

# One distinct colour per class index (1-based)
CLASS_COLORS = {
    1: '#2196F3',   # CoR_circle_ok  — blue
    2: '#F44336',   # CoR_dune_down  — red
    3: '#4CAF50',   # CoR_dune_up    — green
    4: '#FF9800',   # CoR_line       — orange
}
IDX_TO_LABEL = {i + 1: l for i, l in enumerate(config.LABELS)}


def _draw_rect(ax, x1, y1, x2, y2, color, linestyle, linewidth=1.5):
    ax.add_patch(mpatches.Rectangle(
        (x1, y1), x2 - x1, y2 - y1,
        linewidth=linewidth, edgecolor=color,
        facecolor='none', linestyle=linestyle,
    ))


def draw_tile(ax, img_tensor, gt_boxes, gt_labels,
              pred_boxes=None, pred_labels=None, pred_scores=None):
    """
    Render one tile on *ax*.
      GT boxes   — solid lines
      Pred boxes — dashed lines, score shown above box
    """
    arr = img_tensor.permute(1, 2, 0).cpu().numpy()
    gray = (arr[:, :, 0] * 255).astype(np.uint8)
    ax.imshow(gray, cmap='gray', vmin=0, vmax=255)

    for box, lbl in zip(gt_boxes, gt_labels):
        color = CLASS_COLORS.get(int(lbl), '#FFFFFF')
        _draw_rect(ax, *box, color, 'solid')

    if pred_boxes is not None:
        for box, lbl, score in zip(pred_boxes, pred_labels, pred_scores):
            color = CLASS_COLORS.get(int(lbl), '#FFFFFF')
            _draw_rect(ax, *box, color, 'dashed')
            ax.text(box[0], box[1] - 2, f'{score:.2f}',
                    color=color, fontsize=5, va='bottom',
                    bbox=dict(facecolor='black', alpha=0.3, pad=1, linewidth=0))

    ax.axis('off')


def make_legend_handles():
    """Line2D handles for all classes × {GT solid, pred dashed}."""
    handles = []
    for idx, label in IDX_TO_LABEL.items():
        c = CLASS_COLORS.get(idx, '#FFFFFF')
        handles.append(mlines.Line2D([], [], color=c, linewidth=1.5,
                                     linestyle='solid',  label=f'{label} GT'))
        handles.append(mlines.Line2D([], [], color=c, linewidth=1.5,
                                     linestyle='dashed', label=f'{label} pred'))
    return handles


def fig_to_chw_tensor(fig) -> torch.Tensor:
    """Render a matplotlib figure to a (3, H, W) uint8 tensor for TensorBoard."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    from PIL import Image as PILImage
    arr = np.array(PILImage.open(buf).convert('RGB'))
    return torch.from_numpy(arr).permute(2, 0, 1)


def save_prediction_tiles(model, dataset, device, out_dir,
                           n=10, score_threshold=None):
    """
    Save up to *n* tiles (those with GT annotations) showing:
      - GT boxes as solid lines
      - model predictions as dashed lines
    """
    from pathlib import Path
    score_threshold = score_threshold or config.SCORE_THRESHOLD
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    indices = [i for i, t in enumerate(dataset.tiles) if t['labels']][:n]

    model.eval()
    with torch.no_grad():
        for i, idx in enumerate(indices):
            img_t, target = dataset[idx]
            preds = model([img_t.to(device)])[0]
            keep  = preds['scores'] >= score_threshold

            fig, ax = plt.subplots(figsize=(7, 7))
            draw_tile(
                ax, img_t,
                target['boxes'].numpy(), target['labels'].numpy(),
                preds['boxes'][keep].cpu().numpy(),
                preds['labels'][keep].cpu().numpy(),
                preds['scores'][keep].cpu().numpy(),
            )
            tile_name = Path(dataset.tiles[idx]['tile_path']).stem
            ax.set_title(tile_name, fontsize=7)

            legend = make_legend_handles()
            ax.legend(handles=legend, loc='upper right', fontsize=5,
                      framealpha=0.8, ncol=2)

            fig.tight_layout()
            fig.savefig(out_dir / f'pred_{i:02d}.png', dpi=150, bbox_inches='tight')
            plt.close(fig)

    print(f"Saved {len(indices)} prediction images → {out_dir}")


def log_sample_images(writer, tag, model, dataset, device,
                      global_step, n=4, score_threshold=None):
    """Log a row of tiles with GT + predictions to TensorBoard."""
    score_threshold = score_threshold or config.SCORE_THRESHOLD
    indices = [i for i, t in enumerate(dataset.tiles) if t['labels']][:n]
    if not indices:
        return

    fig, axes = plt.subplots(1, len(indices), figsize=(6 * len(indices), 6))
    if len(indices) == 1:
        axes = [axes]

    model.eval()
    with torch.no_grad():
        for ax, idx in zip(axes, indices):
            img_t, target = dataset[idx]
            preds = model([img_t.to(device)])[0]
            keep  = preds['scores'] >= score_threshold
            draw_tile(
                ax, img_t,
                target['boxes'].numpy(), target['labels'].numpy(),
                preds['boxes'][keep].cpu().numpy(),
                preds['labels'][keep].cpu().numpy(),
                preds['scores'][keep].cpu().numpy(),
            )

    handles = make_legend_handles()
    fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=6,
               bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout()

    writer.add_image(tag, fig_to_chw_tensor(fig), global_step)
    plt.close(fig)
