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


def build_model(backbone: str = 'resnet50', pretrained: bool = True, num_classes: int = None):
    if backbone not in BACKBONES:
        raise ValueError(f"Unknown backbone '{backbone}'. Choose from: {list(BACKBONES)}")

    if num_classes is None:
        num_classes = config.NUM_CLASSES

    constructor = BACKBONES[backbone]
    model = constructor(weights='DEFAULT' if pretrained else None)

    # Match anchor tuple count to the actual number of FPN levels in this backbone
    num_levels = len(model.rpn.anchor_generator.sizes)
    sizes = config.ANCHOR_SIZES[:num_levels] if num_levels <= len(config.ANCHOR_SIZES) \
        else config.ANCHOR_SIZES + config.ANCHOR_SIZES[-1:] * (num_levels - len(config.ANCHOR_SIZES))
    aspect_ratios = (config.ANCHOR_ASPECT_RATIOS[0],) * num_levels

    model.rpn.anchor_generator = AnchorGenerator(sizes=sizes, aspect_ratios=aspect_ratios)

    # Replace classification head for our number of classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model
