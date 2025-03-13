import os
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO
import cv2
import numpy as np

class BuildingDataset(Dataset):
    def __init__(self, images_dir, annotation_path, transform=None):
        self.images_dir = images_dir
        self.coco = COCO(annotation_path)
        self.image_ids = list(self.coco.imgs.keys())
        self.transform = transform

        # Filter out image IDs that don't exist on disk
        self.image_ids = [
            img_id for img_id in self.image_ids
            if os.path.exists(os.path.join(self.images_dir, self.coco.imgs[img_id]['file_name']))
        ]
        
        print(f"Filtered dataset: {len(self.image_ids)} valid images found.")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.coco.loadImgs(image_id)[0]
        image_path = os.path.join(self.images_dir, image_info['file_name'])

        image = cv2.imread(image_path)
        
        # Handle missing or corrupt images
        if image is None:
            print(f"Warning: Image {image_path} is missing or corrupt, skipping...")
            return self.__getitem__((idx + 1) % len(self.image_ids))  # Load next valid image

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)

        masks = np.zeros((image_info['height'], image_info['width']), dtype=np.uint8)
        for ann in anns:
            masks = np.maximum(masks, self.coco.annToMask(ann) * ann["category_id"])

        return torch.tensor(image).permute(2, 0, 1), torch.tensor(masks, dtype=torch.uint8)
