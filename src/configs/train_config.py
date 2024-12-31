import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "/content/drive/MyDrive/ProjeDosyalari/rgb_dsm_train/"
VAL_DIR = "/content/drive/MyDrive/ProjeDosyalari/rgb_dsm_val/"
LEARNING_RATE = 2e-4
BATCH_SIZE = 2
NUM_WORKERS = 0
PATCH_SIZE = 512
CHANNELS_IMG = 3
L1_LAMBDA = 100
NUM_EPOCHS = 500
LOAD_MODEL = False
SAVE_MODEL = False
CHECKPOINT_DISC = "disc.pth.tar"
CHECKPOINT_GEN = "gen.pth.tar"
SAVE_EPOCH = 10

both_transform = A.Compose(
    [A.Resize(width=PATCH_SIZE, height=PATCH_SIZE),], additional_targets={"image0": "image"},
)

transform_only_input = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Normalize(mean=[0.5], std=[0.5], max_pixel_value=65535.0,),
        ToTensorV2(),
    ]
)

transform_only_mask = A.Compose(
    [
        A.Normalize(mean=[0.5], std=[0.5], max_pixel_value=65535.0,),
        ToTensorV2(),
    ]
)