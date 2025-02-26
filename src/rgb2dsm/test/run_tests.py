from src.rgb2dsm.training.dataset import MapDataset
import torch
from predict import Predict
import argparse
from calculate_errors import ErrorCalculator
from src.rgb2dsm.training.generator import Generator


parser = argparse.ArgumentParser(description="Test configuration")
parser.add_argument("--version", type=str, required=True, help="Model Version")
args = parser.parse_args()

MODEL_VERSION = args.version

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Load generator weights
generator_path = f"src/rgb2dsm/models/{MODEL_VERSION}/weights/gen.pth.tar"

test_dataset = MapDataset("src/rgb2dsm/datasets/test")
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

predictor = Predict(generator_path,"cpu")

predictor.predict(f"src/rgb2dsm/models/{MODEL_VERSION}/predictions",f"src/rgb2dsm/models/{MODEL_VERSION}/test_results",test_loader)

error_calculator = ErrorCalculator(f"src/rgb2dsm/models/{MODEL_VERSION}/predictions",MODEL_VERSION)

error_calculator.calculate_errors(f"src/rgb2dsm/models/{MODEL_VERSION}/test_results")