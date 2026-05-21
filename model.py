import torchvision
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import config


def build_model(pretrained: bool = True):
    anchor_generator = AnchorGenerator(
        sizes=config.ANCHOR_SIZES,
        aspect_ratios=config.ANCHOR_ASPECT_RATIOS,
    )

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights='DEFAULT' if pretrained else None,
        rpn_anchor_generator=anchor_generator,
    )

    # Swap classification head for our number of classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, config.NUM_CLASSES)

    return model
