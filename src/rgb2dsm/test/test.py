import numpy as np
from PIL import Image
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from src.rgb2dsm.training.generator import Generator
from src.rgb2dsm.training.dataset import MapDataset

class TestClass:
    def __init__(self, model_path, device="cpu"):
        self.device = device
        self.model = self.load_generator(model_path, device)
        self.model.eval()

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

    def test(self, test_loader, save_path="test_results"):
        os.makedirs(save_path, exist_ok=True)
        
        loop = tqdm(test_loader, leave=True)
        metrics = {
            "mse": [],
            "rmse": [],
            "mae": [],
            "ssim": [],
            "psnr": []
        }
        
        with torch.no_grad():
            for idx, (x, y) in enumerate(loop):
                x = x.to(self.device)
                y = y.to(self.device)
                
                pred_dsms = self.model(x)
                pred_np = pred_dsms.cpu().numpy()
                target_np = y.cpu().numpy()

                for pred, target in zip(pred_np, target_np):
                    pred = np.squeeze(pred)
                    target = np.squeeze(target)
                    
                    if np.any(np.isnan(pred)) or np.any(np.isnan(target)) or \
                       np.any(np.isinf(pred)) or np.any(np.isinf(target)):
                        print(f"Warning: NaN or Inf values detected in batch {idx}")
                        continue
                    
                    try:
                        mse = mean_squared_error(target, pred)
                        rmse = np.sqrt(mse)
                        mae = mean_absolute_error(target, pred)
                        
                        metrics['mse'].append(mse)
                        metrics['rmse'].append(rmse)
                        metrics['mae'].append(mae)
                    except Exception as e:
                        print(f"Error calculating basic metrics: {e}")
                        continue
                    
                    try:
                        pred_norm = self.normalize_safe(pred)
                        target_norm = self.normalize_safe(target)
                        
                        if not np.all(pred_norm == 0) and not np.all(target_norm == 0):
                            ssim_score = ssim(target_norm, pred_norm, data_range=1.0)
                            psnr_score = psnr(target_norm, pred_norm, data_range=1.0)
                            
                            metrics['ssim'].append(ssim_score)
                            metrics['psnr'].append(psnr_score)
                    except Exception as e:
                        print(f"Error calculating SSIM/PSNR: {e}")
                """
                if idx % 50 == 0:
                    self.visualize_predictions(x[0], y[0], pred_dsms[0], f'{save_path}/pred_{idx}.png')
                """
                self.save_combined_image(y[0], pred_dsms[0], f'{save_path}/combined_{idx}.png')

        avg_metrics = {k: np.mean(v) for k, v in metrics.items()}

        print("\nEvaluation Results:")
        print(f"Mean Squared Error: {avg_metrics['mse']:.4f}")
        print(f"Root Mean Squared Error: {avg_metrics['rmse']:.4f}")
        print(f"Mean Absolute Error: {avg_metrics['mae']:.4f}")
        
        if 'ssim' in avg_metrics and not np.isnan(avg_metrics['ssim']):
            print(f"Structural Similarity Index (SSIM): {avg_metrics['ssim']:.4f}")
        else:
            print("SSIM: Could not be calculated (possible constant values in images)")
            
        if 'psnr' in avg_metrics and not np.isnan(avg_metrics['psnr']):
            print(f"Peak Signal-to-Noise Ratio (PSNR): {avg_metrics['psnr']:.4f} dB")
        else:
            print("PSNR: Could not be calculated (possible constant values in images)")
        
        return avg_metrics

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

if __name__ == "__main__":
    # Test the model on the test dataset
    test_dataset = MapDataset("src/rgb2dsm/datasets/test")
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    predictor = TestClass("src/rgb2dsm/models/v0.2/weights/gen.pth.tar", "cpu")
    predictor.test(test_loader, "src/rgb2dsm/models/v0.2/testinggg")