import os
import argparse
from elevate3d.core.mesh.generate_mesh import MeshGenerator
from elevate3d.core.dsm.predict_dsm import predict_dsm
from elevate3d.core.dsm.dsm2dtm import generate_dtm
from elevate3d.core.mask.predict_mask import predict_mask
from elevate3d.core.deepforest.deepforest import run_deepforest
import cv2


def run_pipeline(image_path, output_model_path=None):
    """
    Run the entire pipeline for 3D reconstruction.

    Args:
        image_path (str): Path to the input RGB image.
        output_model_path (str): Path to save the resulting 3D model (.glb). If None, show interactively.

    Returns:
        str or None: Path to saved model if output_model_path is provided.
    """
    IMAGE_PATH = image_path
    rgb_image = cv2.imread(IMAGE_PATH)
    if rgb_image is None:
        raise ValueError(f"Image at {IMAGE_PATH} could not be loaded.")
    if rgb_image.shape[:2] != (512, 512):
        actual_size = f"{rgb_image.shape[1]}x{rgb_image.shape[0]}"
        raise ValueError(f"Image must be 512x512 pixels. Actual size: {actual_size}")

    dsm = predict_dsm(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY))
    dtm = generate_dtm(dsm)
    mask = predict_mask(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))
    tree_boxes = run_deepforest(os.path.abspath(IMAGE_PATH))

    mesh_generator = MeshGenerator(rgb_image, dsm, dtm, mask, tree_boxes)

    if output_model_path:
        mesh_generator.visualize(save_path=output_model_path)
        return output_model_path
    else:
        mesh_generator.visualize()
        return None
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the 3D reconstruction pipeline.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input RGB image.")
    parser.add_argument("--output_model_path", type=str, help="Path to save the resulting 3D model (.glb).")
    args = parser.parse_args()

    run_pipeline(args.image_path, args.output_model_path)


        

