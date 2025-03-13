import torch
import torch.nn as nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
NUM_WORKERS = 2
NUM_EPOCHS = 100
