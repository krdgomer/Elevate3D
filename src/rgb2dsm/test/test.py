import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt
from src.training.rgb2dsm.generator import Generator
from src.training.rgb2dsm.dataset import MapDataset
from tqdm import tqdm

def normalize_safe(array):
    """
    Safely normalize array to [0,1] range, handling edge cases
    """
    array_min = np.min(array)
    array_max = np.max(array)
    
    # Check if the array is constant (max == min)
    if array_max == array_min:
        return np.zeros_like(array)  # Return array of zeros if constant
    
    # Normal normalization
    return (array - array_min) / (array_max - array_min)

# Load the Pix2Pix generator model
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

def test(model,test_loader,device,save_path="test_results"):
    loop = tqdm(test_loader, leave=True)
    model.eval()
    metrics ={
        "mse": [],
        "rmse": [],
        "mae": [],
        "ssim": [],
        "psnr": []
    }
    with torch.no_grad():
        for idx,(x,y) in enumerate(loop):

        
            x = x.to(device)
            y = y.to(device)
            
           
            pred_dsms = model(x)
            
            
            pred_np = pred_dsms.cpu().numpy()
            target_np = y.cpu().numpy()

            for pred, target in zip(pred_np, target_np):
            
                pred = np.squeeze(pred)
                target = np.squeeze(target)
                
                # Check for NaN or infinite values
                if np.any(np.isnan(pred)) or np.any(np.isnan(target)) or \
                   np.any(np.isinf(pred)) or np.any(np.isinf(target)):
                    print(f"Warning: NaN or Inf values detected in batch {i}")
                    continue
            
                # Calculate basic metrics
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
                
                
                # Calculate SSIM and PSNR with safe normalization
                try:
                    pred_norm = normalize_safe(pred)
                    target_norm = normalize_safe(target)
                    
                    # Only calculate SSIM and PSNR if normalization was successful
                    if not np.all(pred_norm == 0) and not np.all(target_norm == 0):
                        ssim_score = ssim(target_norm, pred_norm, data_range=1.0)
                        psnr_score = psnr(target_norm, pred_norm, data_range=1.0)
                        
                        metrics['ssim'].append(ssim_score)
                        metrics['psnr'].append(psnr_score)
                except Exception as e:
                    print(f"Error calculating SSIM/PSNR: {e}")
                
            
            if idx % 10 == 0:  
                visualize_predictions(x[0], y[0], pred_dsms[0], 
                                    f'{save_path}/pred_{idx}.png')

    
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

def visualize_predictions(input_img, target_dsm, pred_dsm, save_path):
    """
    Visualize input image, target DSM, and predicted DSM side by side
    """
    plt.figure(figsize=(15, 5))
    
    # Plot input image
    plt.subplot(1, 3, 1)
    plt.imshow(input_img.cpu().numpy().transpose(1, 2, 0))
    plt.title('Input Satellite Image')
    plt.axis('off')
    
    # Plot target DSM
    plt.subplot(1, 3, 2)
    plt.imshow(target_dsm.cpu().numpy().squeeze(), cmap='terrain')
    plt.colorbar(label='Elevation')
    plt.title('Target DSM')
    plt.axis('off')
    
    # Plot predicted DSM
    plt.subplot(1, 3, 3)
    plt.imshow(pred_dsm.cpu().numpy().squeeze(), cmap='terrain')
    plt.colorbar(label='Elevation')
    plt.title('Predicted DSM')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    generator = load_generator("src/models/rgb2dsm/v0.1/rgb_dsm_model/gen.pth.tar")
    test_dataset = MapDataset("src/datasets/rgb_dsm/test")
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    device = torch.device("cude" if torch.cuda.is_available() else "cpu")

    metrics = test(generator, test_loader, device)
