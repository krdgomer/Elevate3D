from test import TestClass
from src.rgb2dsm.training.dataset import MapDataset
import torch

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    generator_path = "src/rgb2dsm/models/v0.2/weights/gen.pth.tar"
    tester = TestClass(generator_path, device)
    
    test_dataset = MapDataset("src/rgb2dsm/datasets/test")
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    metrics = tester.test(test_loader, save_path="src/rgb2dsm/models/v0.2/predictions")