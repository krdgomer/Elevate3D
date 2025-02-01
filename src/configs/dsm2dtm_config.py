import os

# Paths
MODEL_PATH = "\src\models\dsm_dtm_model\dsm_to_dtm.pkl"

# Image Properties
IMAGE_WIDTH = 1024
IMAGE_HEIGHT = 512
HALF_WIDTH = IMAGE_WIDTH // 2  # Since DSM and DTM are side by side

# Machine Learning Parameters
TEST_SIZE = 0.2  # 20% test split
N_ESTIMATORS = 100  # Number of trees in the Random Forest
RANDOM_STATE = 42  # For reproducibility