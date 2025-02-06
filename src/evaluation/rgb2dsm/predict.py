import torch
import torchvision.transforms as transforms
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
from src.training.rgb2dsm.generator import Generator  # Import your Pix2Pix Generator model

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

# Preprocess input image
def preprocess_image(image_path, target_size=(512, 512)):
    img = Image.open(image_path).convert("L")  
    original_size = img.size  

    if original_size[0] < 512 or original_size[1] < 512:
        img = img.resize(target_size, Image.BILINEAR)

    # 🟢 Apply Sharpening
    img = img.filter(ImageFilter.SHARPEN)  

    plt.figure(figsize=(6, 6))
    plt.imshow(img, cmap="gray")
    plt.title("Preprocessed & Sharpened Image")
    plt.axis("off")
    plt.show()

    transform = transforms.Compose([transforms.ToTensor()])
    img_tensor = transform(img).unsqueeze(0)  
    return img_tensor, original_size 

# Perform inference using generator
def predict_dsm(generator, image_tensor, original_size, device="cpu"):
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        output = generator(image_tensor)  # Generate DSM
    
    output = output.squeeze(0).cpu().detach()  
    
    # Resize DSM back to original image size
    output_pil = transforms.ToPILImage()(output.clamp(-1, 1))
    output_resized = output_pil.resize(original_size, Image.BILINEAR)

    return output_resized

# Save or visualize output DSM
def save_dsm(output_image, output_path="output_dsm.png"):
    output_image.save(output_path)
    print(f"DSM saved at {output_path}")

# Main function to run the pipeline
def run_pipeline(model_path, input_image_path, output_dsm_path="output_dsm.png", device="cpu"):
    generator = load_generator(model_path, device)
    input_tensor, original_size = preprocess_image(input_image_path)
    dsm_output = predict_dsm(generator, input_tensor, original_size, device)
    save_dsm(dsm_output, output_dsm_path)

    # Show the final output
    plt.figure(figsize=(6, 6))
    plt.imshow(dsm_output, cmap="viridis")
    plt.colorbar(label="Elevation")
    plt.title("Predicted DSM (Resized Back)")
    plt.axis("off")
    plt.show()

# Example usage
run_pipeline("src/models/rgb2dsm/v0.2/rgb_dsm_model/gen.pth.tar", 
             "src/evaluation/rgb2dsm/test2.png", 
             "src/evaluation/rgb2dsm/output_dsm4.png", 
             device="cpu")
