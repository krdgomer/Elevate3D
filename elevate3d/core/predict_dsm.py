import torch
import numpy as np
from elevate3d.core.models import Generator
import albumentations as A
from albumentations.pytorch import ToTensorV2
from huggingface_hub import hf_hub_download


def load_generator(model_path=None, device="cpu"):
    if model_path is None:
        model_path = hf_hub_download(
            repo_id="krdgomer/elevate3d-weights",
            filename="gen.pth.tar",
            cache_dir="hf_cache"  # Optional: keeps files in a project dir
        )

    print(f"Loading generator model from {model_path}...")
    
    generator = Generator().to(device)  
    checkpoint = torch.load(model_path, map_location=device)

    if "state_dict" in checkpoint:  
        generator.load_state_dict(checkpoint["state_dict"])  
    else:
        generator.load_state_dict(checkpoint)  

    generator.eval()  
    return generator

    

def normalize_safe(array):
    """
    Safely normalize array to [0,1] range, handling edge cases
    """
    array_min = np.min(array)
    array_max = np.max(array)
    
    if array_max == array_min:
        return np.zeros_like(array)
    
    return (array - array_min) / (array_max - array_min)

def predict_dsm(input_img):
    """
    Predict DSM from an image path and return it as an OpenCV image (uint8).
    Args:
        image_path: Path to the input image (reads with OpenCV).
    Returns:
        Predicted DSM as an OpenCV image (shape: H x W, dtype: uint8, range: 0-255).
    """
    print("Predicting DSM...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_generator(device=device)
    model.eval()

    print("Loaded Generator Model")

    # Albumentations transforms
    transform = A.Compose([
        A.Resize(width=512, height=512),
        A.ColorJitter(p=0.2),
        A.Normalize(mean=[0.5], std=[0.5], max_pixel_value=255.0),  # Grayscale
        ToTensorV2(),
    ])

    augmented = transform(image=input_img)
    input_tensor = augmented["image"].to(device)

    with torch.no_grad():
        pred_dsms = model(input_tensor.unsqueeze(0))  # Add batch dimension
        pred_np = pred_dsms.cpu().numpy().squeeze()  # Remove batch dimension

        # Normalize the predicted DSM to [0, 255] for visualization
        pred_norm = (normalize_safe(pred_np) * 255).astype(np.uint8)

        return pred_norm