from PIL import Image
import numpy as np
import torch
from torchvision import transforms as T


class MaskRCNNDataset(torch.utils.data.Dataset):
    def __init__(self, images, masks,images_path,masks_path):
        self.imgs = images
        self.masks = masks
        self.images_path = images_path
        self.masks_path = masks_path

    def __getitem__(self, idx):
        # Load the image and mask
        img = Image.open(self.images_path + self.imgs[idx]).convert("RGB")
        mask = Image.open(self.masks_path + self.masks[idx])
        mask = np.array(mask)

        # Get unique object IDs (excluding background)
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:]  # Exclude background (0)

        # Create masks and bounding boxes for each object
        num_objs = len(obj_ids)
        masks = np.zeros((num_objs, mask.shape[0], mask.shape[1]))
        boxes = []
        for i, obj_id in enumerate(obj_ids):
            masks[i] = mask == obj_id
            pos = np.where(masks[i])
            if pos[0].size == 0 or pos[1].size == 0:  # Skip if no pixels
                continue

            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])

            if xmax > xmin and ymax > ymin:  # Filter invalid boxes
                boxes.append([xmin, ymin, xmax, ymax])

        # Handle cases with no valid boxes
        if len(boxes) == 0:
            boxes = torch.empty((0, 4), dtype=torch.float32)
            labels = torch.empty((0,), dtype=torch.int64)
            masks = torch.empty((0, *mask.shape), dtype=torch.uint8)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.ones((len(boxes),), dtype=torch.int64)
            masks = torch.as_tensor(masks, dtype=torch.uint8)

        # Create the target dictionary
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks

        # Convert the image to a tensor
        img = T.ToTensor()(img)

        return img, target

    def __len__(self):
        return len(self.imgs)
    
def custom_collate(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]

    # Stack images into a single tensor
    images = torch.stack(images, dim=0)

    # Ensure targets are consistent
    for target in targets:
        if "boxes" not in target:
            target["boxes"] = torch.empty((0, 4), dtype=torch.float32)
        if "labels" not in target:
            target["labels"] = torch.empty((0,), dtype=torch.int64)
        if "masks" not in target:
            target["masks"] = torch.empty((0, images.shape[2], images.shape[3]), dtype=torch.uint8)

    return images, targets