import torch
import torchvision.transforms as T
import cv2
import numpy as np
from elevate3d.core.models import get_maskrcnn_model
from torchvision.transforms import functional as F
from elevate3d.utils.download_manager import DownloadManager

def post_process(mask):
    mask = (mask > 0).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros_like(mask)

    biggest = max(contours, key=cv2.contourArea)
    epsilon = 0.01 * cv2.arcLength(biggest, True)
    approx = cv2.approxPolyDP(biggest, epsilon, True)

    clean = np.zeros_like(mask)
    cv2.fillPoly(clean, [approx], 255)
    return clean // 255

def predict_mask(input_image):
    """Predict the mask from an input image using a pre-trained Mask R-CNN model.

    Args:
        input_image : Input image as a NumPy array (H x W x C, dtype: uint8).

    Returns:
        labeled_mask: Labeled mask as a NumPy array (H x W, dtype: uint8) with unique labels for each building.
    """
    print("Predicting mask...")

    # Load model
    model = get_maskrcnn_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    download_manager = DownloadManager()
    weights_path = download_manager.download_file("maskrcnn_weights.pth")


    # Load weights
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    # Normalize and convert image to tensor
    image_tensor = F.to_tensor(input_image).unsqueeze(0).to(device)

    # Run model prediction
    with torch.no_grad():
        prediction = model(image_tensor)[0]

    # Extract masks and scores
    masks = (prediction["masks"].squeeze(1) > 0.5).cpu().numpy()
    scores = prediction["scores"].cpu().numpy()

    filtered_indices = np.where(scores >= 0.5)[0]
    filtered_masks = masks[filtered_indices]
    filtered_scores = scores[filtered_indices]

    # Post-process EACH mask individually
    processed_masks = np.array([post_process(mask) for mask in filtered_masks])

    
    from skimage.measure import label

    # Step 1: Combine all binary masks into one (1 = any building, 0 = background)
    combined_mask = np.sum(processed_masks, axis=0) > 0  # Shape [512, 512]

    # Step 2: Label connected regions (8-connectivity to detect diagonal touches)
    instance_segmentation = label(combined_mask, connectivity=2)  # 2 for 8-connectivity

    # Step 3 (Optional): Remove small regions (e.g., <10 pixels)
    from skimage.morphology import remove_small_objects
    cleaned_instance_segmentation = remove_small_objects(
        instance_segmentation, 
        min_size=10, 
        connectivity=2
    )

    return cleaned_instance_segmentation