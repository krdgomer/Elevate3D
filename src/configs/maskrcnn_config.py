import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 0.005
MOMENTUM= 0.9
WEIGHT_DECAY = 0.0005
BATCH_SIZE = 4
NUM_WORKERS = 8
NUM_EPOCHS = 30
LOAD_MODEL = False
SAVE_MODEL = True