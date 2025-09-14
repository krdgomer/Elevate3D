import os
import argparse
from elevate3d.core.mesh.generate_mesh import MeshGenerator
from elevate3d.core.dsm.predict_dsm import predict_dsm
from elevate3d.core.dsm.dsm2dtm import generate_dtm
from elevate3d.core.mask.predict_mask import predict_mask
from elevate3d.core.deepforest.deepforest import run_deepforest
import cv2
import logging

class Pipeline:
    def __init__(self,image_path, output_model_path=None):
        self.image_path = image_path
        self.output_model_path = output_model_path

    def load_image(self):
        logging.info("Loading image...")
        rgb_image = cv2.imread(self.image_path)
        if rgb_image is None:
            raise ValueError(f"Image at {self.image_path} could not be loaded.")
        if rgb_image.shape[:2] != (512, 512):
            actual_size = f"{rgb_image.shape[1]}x{rgb_image.shape[0]}"
            raise ValueError(f"Image must be 512x512 pixels. Actual size: {actual_size}")
        
        return rgb_image

    def process_dsm(self,rgb_image):
        logging.info("Processing DSM...")
        dsm = predict_dsm(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY))
        return dsm
    
    def process_dtm(self,dsm):
        logging.info("Processing DTM...")
        dtm = generate_dtm(dsm)
        return dtm

    def process_mask(self, rgb_image):
        logging.info("Processing Mask...")
        mask = predict_mask(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))
        return mask
    
    def process_tree_boxes(self):
        logging.info("Processing Tree Boxes...")
        tree_boxes = run_deepforest(os.path.abspath(self.image_path))
        return tree_boxes

    def generate_mesh(self, rgb_image, dsm, dtm, mask,tree_boxes,output_model_path=None):
        logging.info("Generating Mesh...")
        mesh_generator = MeshGenerator(rgb_image,dsm,dtm,mask,tree_boxes)

        if output_model_path:
            mesh_generator.visualize(save_path=output_model_path)
            return output_model_path
        else:
            mesh_generator.visualize()
            return None
        
    def run(self):
        rgb_image = self.load_image()
        dsm = self.process_dsm(rgb_image)
        dtm = self.process_dtm(dsm)
        mask = self.process_mask(rgb_image)
        tree_boxes = self.process_tree_boxes()
        result_path = self.generate_mesh(rgb_image, dsm, dtm, mask, tree_boxes, self.output_model_path)
        
        if result_path:
            logging.info(f"3D model saved at: {result_path}")




    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the 3D reconstruction pipeline.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input RGB image.")
    parser.add_argument("--output_model_path", type=str, help="Path to save the resulting 3D model (.glb).")
    args = parser.parse_args()

    pipeline = Pipeline(args.image_path, args.output_model_path)
    pipeline.run()


        

