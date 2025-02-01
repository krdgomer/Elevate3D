import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from src.training.rgb2dsm.generator import Generator  # Import your Pix2Pix Generator model

# Load the Pix2Pix generator model
def load_generator(model_path, device="cpu"):
    print(f"Loading generator model from {model_path}...")
    
    # Initialize the generator model
    generator = Generator().to(device)  # Ensure you have the correct Generator model
    checkpoint = torch.load(model_path, map_location=device)

    if "state_dict" in checkpoint:  
        generator.load_state_dict(checkpoint["state_dict"])  # Load weights
    else:
        generator.load_state_dict(checkpoint)  # Directly load if it's just state_dict

    generator.eval()  # Set to evaluation mode
    return generator

# Preprocess input image
def preprocess_image(image_path, target_size=(512, 512)):
    img = Image.open(image_path).convert("L")  # Gri tonlama
    transform = transforms.Compose([
        transforms.Resize((512, 512)),  # Görüntüyü 512x512'ye ayarla
        transforms.ToTensor(),
    ])
    img_tensor = transform(img).unsqueeze(0)  # (1, 1, 512, 512) şekline getir
    return img_tensor

# Perform inference using generator
def predict_dsm(generator, image_tensor, device="cpu"):
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        output = generator(image_tensor)  # Generate DSM
    output = output.squeeze(0).cpu().detach()  # Remove batch dimension
    return output

# Save or visualize output DSM
def save_dsm(output_tensor, output_path="output_dsm.png"):
    output_image = transforms.ToPILImage()(output_tensor.clamp(-1, 1))  # Clamp values and convert to image
    output_image.save(output_path)
    print(f"DSM saved at {output_path}")

# Main function to run the pipeline
def run_pipeline(model_path, input_image_path, output_dsm_path="output_dsm.png", device="cpu"):
    generator = load_generator(model_path, device)
    input_tensor = preprocess_image(input_image_path)
    dsm_output = predict_dsm(generator, input_tensor, device)
    save_dsm(dsm_output, output_dsm_path)

    # Show the result
    plt.figure(figsize=(6, 6))
    plt.imshow(dsm_output.permute(1, 2, 0).squeeze(), cmap="viridis")
    plt.colorbar(label="Elevation")
    plt.title("Predicted DSM")
    plt.axis("off")
    plt.show()

# Example usage
run_pipeline("src/models/rgb_dsm_model/gen.pth.tar", "src/evaluation/rgb2dsm/test2.png", "src/evaluation/rgb2dsm/output_dsm.png", device="cpu")
