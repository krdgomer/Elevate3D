import torch
import torchvision
from torch.utils.data import Dataset
from pycocotools.coco import COCO
import json
import cv2
import numpy as np

class BuildingDataset(Dataset):
    def __init__(self, root_dir, annotation_path, transform=None):
        self.root_dir = root_dir
        self.coco = COCO(annotation_path)
        self.image_ids = list(self.coco.imgs.keys())
        self.transform = transform
    def __len__(self):
        return len(self.image_ids)
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.coco.loadImgs(image_id)[0]
        image = cv2.imread(f"{self.image_dir}/{image_info['file_name']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)

        masks = np.zeros((image_info['height'], image_info['width']), dtype=np.uint8)
        for ann in anns:
            masks = np.maximum(masks, self.coco.annToMask(ann) * ann["category_id"])

        return torch.tensor(image).permute(2, 0, 1), torch.tensor(masks, dtype=torch.uint8)