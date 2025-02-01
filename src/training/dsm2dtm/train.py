import os
import numpy as np
import joblib
import argparse
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from src.configs.dsm2dtm_config import MODEL_PATH, TEST_SIZE, N_ESTIMATORS, RANDOM_STATE
from preprocess import load_and_split_combined_image, interpolate_missing_data, extract_features

# Argument parser
parser = argparse.ArgumentParser(description="Training Configuration")
parser.add_argument("--dataset_dir", type=str, required=True, help="Path to the dataset directory")
args = parser.parse_args()

DATASET_DIR = args.dataset_dir

def load_dataset():
    """Load and preprocess all images in the dataset directory."""
    features_list, target_list = [], []

    for file_name in os.listdir(DATASET_DIR):
        if not file_name.lower().endswith(".png"):
            continue
        
        file_path = os.path.join(DATASET_DIR, file_name)
        try:
            dsm, dtm = load_and_split_combined_image(file_path)
            dtm = interpolate_missing_data(dtm)

            # Prepare features and target
            features = extract_features(dsm)
            target = dtm.flatten()
            features = features.reshape(-1, features.shape[-1])

            # Remove NaN values
            valid_mask = ~np.isnan(target) & ~np.isnan(features).any(axis=1)
            features_list.append(features[valid_mask])
            target_list.append(target[valid_mask])
        
        except ValueError as e:
            print(f"Skipping {file_name} due to error: {e}")

    return np.vstack(features_list), np.hstack(target_list)

def train_model():
    """Train a Random Forest model and save it."""
    print("Loading dataset...")
    features, target = load_dataset()

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    print("Training Random Forest model...")
    model = RandomForestRegressor(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"RMSE: {rmse:.2f}")

    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_model()
