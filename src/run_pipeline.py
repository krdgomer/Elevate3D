import os
import shutil
import argparse
from pipeline.generate_mesh import MeshGenerator
from pipeline.predict_dsm import predict_dsm
from models.dsm2dtm import generate_dtm
from pipeline.predict_mask import predict_mask
from PIL import Image

if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description="Training Configuration")
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Image path",
    )

    args = parser.parse_args()

    IMAGE_PATH = args.image_path

    # Check if the image is 512x512
    with Image.open(IMAGE_PATH) as img:
        if img.size != (512, 512):
            print("Error: The input image must be 512x512 in size.")
            exit(1)

    # Create the src/temp folder
    os.makedirs("temp", exist_ok=True)

    try:
        # Predict DSM
        predict_dsm(IMAGE_PATH, "temp", "dsm.png")
        
        # Generate DTM
        generate_dtm("temp/dsm.png", "temp/dtm.png")

        # Generate Mask
        predict_mask(IMAGE_PATH, "temp/labeled_mask.png")

        # Generate Mesh
        mesh_generator = MeshGenerator(IMAGE_PATH, "temp/dsm.png", "temp/dtm.png", "temp/labeled_mask.png")
        mesh_generator.generate_terrain_mesh()
    finally:
        # Delete the src/temp folder
        shutil.rmtree("temp")
        

