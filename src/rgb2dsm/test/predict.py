import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from src.rgb2dsm.training.generator import Generator
import os


class Predict():
    def __init__(self,model_path,device):
        self.device = device
        self.model = self.load_generator(model_path, device)
        self.model.eval()
    
    @staticmethod
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
        
    @staticmethod
    def normalize_safe(array):
        """
        Safely normalize array to [0,1] range, handling edge cases
        """
        array_min = np.min(array)
        array_max = np.max(array)
        
        if array_max == array_min:
            return np.zeros_like(array)
        
        return (array - array_min) / (array_max - array_min)
    
    def predict(self,predictions_save_dir,tests_save_dir,test_loader):
        os.makedirs(predictions_save_dir, exist_ok=True)
        os.makedirs(tests_save_dir, exist_ok=True)
        loop = tqdm(test_loader, leave=True)

        with torch.no_grad():
            for idx, (input_img,target_img) in enumerate(loop):
                input_img = input_img.to(self.device)
                target_img = target_img.to(self.device)

                pred_dsms = self.model(input_img)
                pred_np = pred_dsms.cpu().numpy()
                target_np = target_img.cpu().numpy()

                if idx % 50 == 0:   
                    self.visualize_predictions(input_img[0], target_img[0], pred_dsms[0], f'{tests_save_dir}/sample_{idx}.png')

                self.save_combined_image(target_img[0], pred_dsms[0], f'{predictions_save_dir}/prediction_{idx}.png') 

    def save_combined_image(self, input_img, pred_dsm, save_path):
        """
        Save combined image with input image on the left and predicted DSM on the right.
        No axes, titles, or colorbars are included.
        """
        # Convert tensors to numpy arrays
        input_np = input_img.cpu().numpy().transpose(1, 2, 0)  # Convert to HWC format
        pred_np = pred_dsm.cpu().numpy().squeeze()  # Remove batch and channel dimensions

        # Normalize the predicted DSM to [0, 255] for visualization
        pred_norm = self.normalize_safe(pred_np) * 255
        pred_norm = pred_norm.astype(np.uint8)

        # Convert input image to uint8 if it's not already
        if input_np.dtype != np.uint8:
            input_np = (self.normalize_safe(input_np) * 255).astype(np.uint8)

        # Create a blank canvas to combine the images
        height = max(input_np.shape[0], pred_norm.shape[0])
        width = input_np.shape[1] + pred_norm.shape[1]
        combined_image = np.zeros((height, width, 3), dtype=np.uint8)

        # Place the input image on the left
        combined_image[:input_np.shape[0], :input_np.shape[1], :] = input_np

        # Place the predicted DSM on the right (convert grayscale to RGB)
        pred_rgb = np.stack([pred_norm] * 3, axis=-1)  # Convert grayscale to RGB
        combined_image[:pred_rgb.shape[0], input_np.shape[1]:, :] = pred_rgb

        # Save the combined image using PIL
        combined_pil = Image.fromarray(combined_image)
        combined_pil.save(save_path)

    def visualize_predictions(self, input_img, target_dsm, pred_dsm, save_path):
        """
        Visualize input image, target DSM, and predicted DSM side by side
        """
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.imshow(input_img.cpu().numpy().transpose(1, 2, 0))
        plt.title('Input Satellite Image')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(target_dsm.cpu().numpy().squeeze(), cmap='gray')
        plt.colorbar(label='Elevation')
        plt.title('Target DSM')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(pred_dsm.cpu().numpy().squeeze(), cmap='gray')
        plt.colorbar(label='Elevation')
        plt.title('Predicted DSM')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()