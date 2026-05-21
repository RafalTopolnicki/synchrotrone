from pathlib import Path
import torch

ROOT = Path(__file__).parent
DATA_DIR = ROOT / 'DATA'
ANNOTATIONS_DIR = DATA_DIR / 'annotations'
IMAGES_DIR = DATA_DIR
TILES_DIR = DATA_DIR / 'tiles'
SPLITS_FILE = DATA_DIR / 'splits.json'
CHECKPOINTS_DIR = ROOT / 'checkpoints'

LABELS = ['CoR_circle_ok', 'CoR_dune_down', 'CoR_dune_up', 'CoR_line']
NUM_CLASSES = len(LABELS) + 1  # +1 for background (class 0)

# Tiling — 640px tiles with 128px overlap (stride=512)
TILE_SIZE = 640
OVERLAP = 128
STRIDE = TILE_SIZE - OVERLAP  # 512
# Minimum fraction of a box's area that must fall inside a tile to keep it
MIN_BBOX_OVERLAP = 0.3

# Train/val/test split (at base-image level, so _sat variants follow their base)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

# Training
BATCH_SIZE = 4
NUM_EPOCHS = 30
LR = 0.005
LR_MOMENTUM = 0.9
LR_WEIGHT_DECAY = 0.0005
LR_STEP_SIZE = 10
LR_GAMMA = 0.1
NUM_WORKERS = 4

# Anchors tuned for small objects (median ~25px on 640px tiles)
ANCHOR_SIZES = ((8,), (16,), (32,), (64,), (128,))
ANCHOR_ASPECT_RATIOS = ((0.5, 1.0, 2.0),) * 5

# Inference
SCORE_THRESHOLD = 0.5
NMS_IOU_THRESHOLD = 0.4

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
