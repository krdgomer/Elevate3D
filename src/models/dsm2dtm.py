import cv2
import numpy as np
import rasterio
from rasterio.transform import from_origin
from bulldozer.pipeline.bulldozer_pipeline import dsm_to_dtm
from PIL import Image

def generate_dtm(dsm_path, output_path, kernel_size=49, smooth_sigma=10):
    # Load DSM image
    dsm = Image.open(dsm_path)

    # Create a black image with same size
    dtm = Image.new("L", dsm.size, 0)

    # Save as PNG
    dtm.save(output_path)