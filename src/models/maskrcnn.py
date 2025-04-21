import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

def get_model():
    """Load a pre-trained Mask R-CNN model from torchvision and modify it for binary classification.
    The model is based on a ResNet-50 backbone with a Feature Pyramid Network (FPN) for improved feature extraction.

    Returns:
        model: A Mask R-CNN model with a ResNet-50 backbone and a Feature Pyramid Network (FPN) for improved feature extraction.
    """
    model = torchvision.models.detection.maskrcnn_resnet50_fpn()
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features , 2)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask , hidden_layer , 2)
    return model