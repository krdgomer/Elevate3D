import numpy as np
from src.configs import train_config as config
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
import cv2


class MapDataset(Dataset):
    def __init__(self, root_dir, apply_histogram_eq=False):
        self.root_dir = root_dir
        self.rgb_dir = os.path.join(root_dir, "rgb")
        self.dsm_dir = os.path.join(root_dir, "dsm")
        self.image_names = [f for f in os.listdir(self.rgb_dir) if f.endswith('.png')]
        self.apply_histogram_eq = apply_histogram_eq

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        img_name = self.image_names[index]
        
        rgb_path = os.path.join(self.rgb_dir, img_name)
        dsm_path = os.path.join(self.dsm_dir, img_name)

        input_image = np.array(Image.open(rgb_path))
        target_image = np.array(Image.open(dsm_path))
        
        input_image = np.array(Image.fromarray(input_image).convert("L"))
        target_image = np.array(Image.fromarray(input_image).convert("L"))

        # Apply histogram equalization if enabled
        if self.apply_histogram_eq:
            input_image = cv2.equalizeHist(input_image)

        # Apply augmentations
        augmentations = config.both_transform(image=input_image, image0=target_image)
        input_image = augmentations["image"]
        target_image = augmentations["image0"]

        input_image = config.transform_only_input(image=input_image)["image"]
        target_image = config.transform_only_mask(image=target_image)["image"]

        return input_image, target_image


if __name__ == "__main__":
    dataset = MapDataset("data/train/", apply_histogram_eq=True)
    loader = DataLoader(dataset, batch_size=5)
    for x, y in loader:
        print(x.shape)
        save_image(x, "x.png")
        save_image(y, "y.png")
        import sys

        sys.exit()
