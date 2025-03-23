import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import torch
import torchvision.transforms as T
import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.BATC.model import get_model

if __name__ == "__main__":
    # Load the trained model
    model = get_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the trained weights
    model.load_state_dict(torch.load("src/BATC/models/v0.0/maskrcnn_weights.pth", map_location=device))
    model.eval()

    image_path = "src/BATC/test/input2.png"
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    transform = T.Compose([T.ToTensor()])
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        predictions = model(input_tensor)

    # Get masks and scores
    masks = predictions[0]['masks'].squeeze().detach().cpu().numpy()
    scores = predictions[0]['scores'].detach().cpu().numpy()
    threshold = 0.5

    # Create an empty labeled mask image
    labeled_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

    # Assign unique values in descending order (255, 254, 253, ...)
    label_value = 255

    for i in range(len(masks)):
        if scores[i] > threshold:
            mask = (masks[i] > 0.5).astype(np.uint8)

            # Apply Morphological Operations (Erosion + Dilation)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # Adjust size if needed
            mask = cv2.erode(mask, kernel, iterations=1)
            mask = cv2.dilate(mask, kernel, iterations=2)

            # Assign a unique label
            labeled_mask[mask == 1] = label_value
            
            # Decrease label value (but not below 1)
            label_value = max(1, label_value - 1)

    # Save the labeled mask
    output_path = "src/BATC/test/labeled_mask.png"
    cv2.imwrite(output_path, labeled_mask)

    # Show the result
    plt.figure(figsize=(10, 10))
    plt.imshow(labeled_mask, cmap="gray")
    plt.axis("off")
    plt.title("Labeled Building Mask")
    plt.show()

    print(f"Labeled mask saved to {output_path}")
