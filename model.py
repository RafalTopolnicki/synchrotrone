import torchvision
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import config

# Supported backbone names → torchvision constructor
BACKBONES = {
    'resnet50':   torchvision.models.detection.fasterrcnn_resnet50_fpn,
    'resnet50v2': torchvision.models.detection.fasterrcnn_resnet50_fpn_v2,
    'mobilenet':  torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn,
}


def build_model(backbone: str = 'resnet50', pretrained: bool = True):
    if backbone not in BACKBONES:
        raise ValueError(f"Unknown backbone '{backbone}'. Choose from: {list(BACKBONES)}")

    anchor_generator = AnchorGenerator(
        sizes=config.ANCHOR_SIZES,
        aspect_ratios=config.ANCHOR_ASPECT_RATIOS,
    )

    constructor = BACKBONES[backbone]
    model = constructor(weights='DEFAULT' if pretrained else None)
    # Override anchor generator after construction so it works for all backbones
    # (some constructors set rpn_anchor_generator internally, causing conflicts)
    model.rpn.anchor_generator = anchor_generator

    # Replace classification head for our number of classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, config.NUM_CLASSES)

    return model
