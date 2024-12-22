import numpy as np
from configs import train_config as config
import os
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
import cv2


class MapDataset(Dataset):
    def __init__(self, root_dir, apply_histogram_eq=False):
        self.root_dir = root_dir
        self.list_files = os.listdir(self.root_dir)
        self.apply_histogram_eq = apply_histogram_eq

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, index):
        img_file = self.list_files[index]
        img_path = os.path.join(self.root_dir, img_file)
        image = np.array(Image.open(img_path))

        # Split into input and target images
        input_image = image[:, :512, :]
        target_image = image[:, 512:, :]

        # Convert both images to grayscale
        input_image = np.array(Image.fromarray(input_image).convert("L"))
        target_image = np.array(Image.fromarray(target_image).convert("L"))

        # Apply histogram equalization if enabled
        if self.apply_histogram_eq:
            input_image = cv2.equalizeHist(input_image)
            target_image = cv2.equalizeHist(target_image)

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
