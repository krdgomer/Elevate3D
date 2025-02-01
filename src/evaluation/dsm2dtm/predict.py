import joblib
import numpy as np
import cv2
import matplotlib.pyplot as plt
from src.configs.dsm2dtm_config import MODEL_PATH
from preprocess import load_and_split_combined_image, extract_features
from src.utils.utils import save_image

def predict_dtm(image_path):
    """Predict DTM from a given DSM using the trained model."""
    model = joblib.load(MODEL_PATH)
    dsm, _ = load_and_split_combined_image(image_path)  # Use DSM, ignore DTM

    # Extract features
    input_features = extract_features(dsm)
    input_features = input_features.reshape(-1, input_features.shape[-1])

    # Predict
    predicted_dtm = model.predict(input_features)
    predicted_dtm = predicted_dtm.reshape(dsm.shape)

    # Save and visualize
    output_path = "predicted_dtm.png"
    save_image(predicted_dtm, output_path)
    print(f"Predicted DTM saved to {output_path}")

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Original DSM")
    plt.imshow(dsm, cmap="terrain")
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.title("Predicted DTM")
    plt.imshow(predicted_dtm, cmap="terrain")
    plt.colorbar()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_image = "path_to_test_combined_image.png"  # Change this path
    predict_dtm(test_image)
