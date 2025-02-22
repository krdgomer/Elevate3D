import torch
import torchvision
from torch.utils.data import Dataset

class BuildingDataset(Dataset):
    def __init__(self, root_dir, annotation_path, transform=None):
        self.root_dir = root_dir
        self.coco = COCO(annotation_path)
        self.image_ids = list(self.coco.imgs.keys())
        self.transform = transform
    def __len__(self):
        return len(self.image_ids)
    def __getitem__(self, idx):
        pass