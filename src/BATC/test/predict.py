import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
from model import get_model  # Import the model definition

# Load the trained model
model = get_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load the trained weights
model.load_state_dict(torch.load("model_weights.pth", map_location=device))
model.eval()

def visualize_prediction(image_path):
    img = Image.open(image_path).convert("RGB")
    transform = T.ToTensor()
    img_tensor = transform(img).to(device)
    
    with torch.no_grad():
        pred = model([img_tensor])
    
    plt.imshow(img)
    for mask in pred[0]["masks"]:
        plt.imshow(mask.cpu().squeeze(), alpha=0.5, cmap="jet")
    plt.show()

# Example usage
image_path = "test.jpg"
visualize_prediction(image_path)