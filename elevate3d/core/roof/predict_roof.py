import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from elevate3d.utils.download_manager import DownloadManager

# Simple Roof Type Predictor
class SimpleRoofPredictor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Define class names (should match your training)
        self.class_names = ['complex', 'flat', 'gable', 'hip', 'pyramid']
        
        # Load model
        download_manager = DownloadManager()
        model_path = download_manager.download_file("roof_classifier_improved.pth")
        self.model = self.load_model(model_path)
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(235),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def load_model(self, model_path):
        """Load the trained model"""
        # Create model architecture (EfficientNet-B0 like during training)
        model = models.efficientnet_b0(weights=None)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(num_ftrs, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, len(self.class_names))
        )
        
        # Load trained weights
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        
        print("Model loaded successfully!")
        return model
    
    def predict(self, image):
        """Predict roof type for an image"""
        try:
            # Ensure the image is in RGB format
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            # Load and process image
            input_tensor = self.transform(image)
            input_batch = input_tensor.unsqueeze(0).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                output = self.model(input_batch)
                probabilities = torch.softmax(output, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)
            
            # Get results
            predicted_class = self.class_names[predicted_idx.item()]
            confidence_score = confidence.item()
            all_probs = probabilities[0].cpu().numpy()
            
            return predicted_class, confidence_score, all_probs, image
            
        except Exception as e:
            print(f"Error: {e}")
            return None, None, None, None

