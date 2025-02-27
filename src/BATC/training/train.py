import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import matplotlib.pyplot as plt
import numpy as np
import cv2
from src.BATC.training.dataset import BuildingsDataset
import argparse
import src.configs.maskrcnn_config as cfg

parser = argparse.ArgumentParser(description="Training Configuration")
parser.add_argument("--images_dir", type=str, required=True)
parser.add_argument("--annotations_dir", type=str, required=True)
args = parser.parse_args()

IMAGES_DIR = args.images_dir
ANNOTATIONS_DIR = args.annotations_dir

dataset = BuildingsDataset(IMAGES_DIR, ANNOTATIONS_DIR)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=cfg.NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS)

def get_model(num_classes):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = MaskRCNNPredictor(in_features, num_classes)

    return model

num_classes = 2  # Background + Buildings
model = get_model(num_classes)

device = cfg.DEVICE
model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

for epoch in range(cfg.NUM_EPOCHS):
    model.train()
    total_loss = 0

    for images, targets in train_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    lr_scheduler.step()

    print(f"Epoch {epoch+1}/{cfg.NUM_EPOCHS}, Loss: {total_loss:.4f}")

torch.save(model.state_dict(), "mask_rcnn_buildings.pth")
    
model.eval()
with torch.no_grad():
    for images, targets in val_loader:
        images = [img.to(device) for img in images]
        predictions = model(images)

        # Visualize Predictions
        for img, pred in zip(images, predictions):
            img = img.permute(1, 2, 0).cpu().numpy()
            plt.figure(figsize=(10, 10))
            plt.imshow(img)

            for mask in pred["masks"]:
                mask = mask[0].cpu().numpy()
                plt.imshow(mask, alpha=0.5, cmap="jet")

            plt.show()