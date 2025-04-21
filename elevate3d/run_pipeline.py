import os
import argparse
from pipeline.generate_mesh import MeshGenerator
from pipeline.predict_dsm import predict_dsm
from models.dsm2dtm import generate_dtm
from pipeline.predict_mask import predict_mask
from pipeline.deepforest import run_deepforest
import cv2

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

    #Load image into memory
    rgb_image = cv2.imread(IMAGE_PATH)
    if rgb_image is None:
        raise ValueError(f"Image at {IMAGE_PATH} could not be loaded. Please check the path and format.")
    # Check dimensions (OpenCV uses height-first: shape = (h, w) or (h, w, channels))
    if rgb_image.shape[:2] != (512, 512):
        actual_size = f"{rgb_image.shape[1]}x{rgb_image.shape[0]}"  # Format as "width x height"
        raise ValueError(f"Image must be 512x512 pixels. Actual size: {actual_size}. ")

    # Predict DSM
    dsm = predict_dsm(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY))
    
    # Generate DTM
    dtm= generate_dtm(dsm)

    # Generate Mask
    mask = predict_mask(cv2.cvtColor(rgb_image,cv2.COLOR_BGR2RGB))

    tree_boxes = run_deepforest(os.path.abspath(IMAGE_PATH))

    # Generate Mesh
    mesh_generator = MeshGenerator(rgb_image, dsm, dtm, mask ,tree_boxes)
    mesh_generator.visualize()

        

