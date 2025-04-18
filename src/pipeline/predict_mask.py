import torch
import torchvision.transforms as T
import cv2
import numpy as np
from models.maskrcnn import get_model
from skimage.measure import label
import os

def clean_mask(mask):
    """Clean and straighten the mask using morphological operations and polygon approximation."""
    # Ensure binary
    mask = (mask > 0).astype(np.uint8) * 255

    # Morphological closing to fill gaps and smooth noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros_like(mask)

    # Take largest contour
    biggest = max(contours, key=cv2.contourArea)

    # Approximate polygon
    epsilon = 0.01 * cv2.arcLength(biggest, True)
    approx = cv2.approxPolyDP(biggest, epsilon, True)

    # Fill the polygon
    clean = np.zeros_like(mask)
    cv2.fillPoly(clean, [approx], 255)
    return clean // 255  # convert back to binary

def predict_mask(image_path, output_path):
    print("Predicting mask...")

    # Load the trained model
    model = get_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load weights
    model.load_state_dict(torch.load("src/models/weights/maskrcnn_weights.pth", map_location=device))
    model.eval()

    # Load and preprocess image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = T.Compose([T.ToTensor()])
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        predictions = model(input_tensor)

    masks = predictions[0]['masks'].squeeze().detach().cpu().numpy()
    scores = predictions[0]['scores'].detach().cpu().numpy()
    threshold = 0.5

    labeled_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    label_value = 255

    for i in range(len(masks)):
        if scores[i] > threshold:
            raw_mask = (masks[i] > 0.5).astype(np.uint8)

            # Clean and regularize the shape
            cleaned_mask = clean_mask(raw_mask)

            # Label connected components
            labeled_components, num_buildings = label(cleaned_mask, connectivity=2, return_num=True)
            instance_mask = labeled_components.astype(np.uint16)

            # Apply to final output
            labeled_mask[instance_mask > 0] = label_value
            label_value = max(1, label_value - 1)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, labeled_mask)
    print(f"Labeled instance mask saved to {output_path}")