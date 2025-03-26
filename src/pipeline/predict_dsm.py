import torch
from PIL import Image
import numpy as np
from models.generator import Generator
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2

def load_generator(model_path, device="cpu"):
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

def predict_dsm(image_path, predictions_save_dir, save_name="prediction.png"):
    print("Predicting DSM...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_generator("src/models/weights/gen.pth.tar", device)
    model.eval()

    print("Loaded Generator Model")

    os.makedirs(predictions_save_dir, exist_ok=True)

    # Define transformations
    both_transform = A.Compose(
        [A.Resize(width=512, height=512)], additional_targets={"image0": "image"}
    )

    transform_only_input = A.Compose(
        [
            A.ColorJitter(p=0.2),
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0),
            ToTensorV2(),
        ]
    )

    # Load and transform the input image
    input_img = np.array(Image.open(image_path).convert('L'))
    transform = A.Compose([
        both_transform,
        transform_only_input
    ])
    augmented = transform(image=input_img)
    input_img = augmented["image"]
    input_img = input_img.to(device)

    with torch.no_grad():
        pred_dsms = model(input_img.unsqueeze(0))  # Add batch dimension
        pred_np = pred_dsms.cpu().numpy().squeeze()  # Remove batch dimension

        # Normalize the predicted DSM to [0, 255] for visualization
        pred_norm = normalize_safe(pred_np) * 255
        pred_norm = pred_norm.astype(np.uint8)

        # Save the predicted DSM using PIL
        pred_pil = Image.fromarray(pred_norm)
        pred_pil.save(os.path.join(predictions_save_dir, save_name))