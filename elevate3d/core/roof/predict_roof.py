import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from elevate3d.utils.download_manager import DownloadManager
import logging

class SimpleRoofPredictor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {self.device}")

        # Define class names for roof prediction
        self.class_names = ['complex', 'flat', 'gable', 'hip', 'pyramid']

        download_manager = DownloadManager()
        model_path = download_manager.download_file("best_roof_model.pth")
        self.model = self.load_model(model_path)

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    def load_model(self, model_path):
        """Load the trained ResNet50 with custom classifier"""
        num_classes = len(self.class_names)
        model = models.resnet50(pretrained=True)

        # Freeze all layers except layer4 and fc
        for name, param in model.named_parameters():
            param.requires_grad = False
            if "layer4" in name or "fc" in name:
                param.requires_grad = True

        # Replace classifier head with the target number of roof classes
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

        # Load trained weights
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model = model.to(self.device)
        model.eval()
        logging.info("Model loaded successfully!")
        return model

    def predict(self, image):
        """Predict roof type for a single image file"""
        if image.mode != "RGB":
            image = image.convert("RGB")
        input_tensor = self.transform(image)
        input_batch = input_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(input_batch)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)

        predicted_class = self.class_names[predicted_idx.item()]
        confidence_score = confidence.item()
        all_probs = probabilities[0].cpu().numpy()

        return predicted_class, confidence_score, all_probs
