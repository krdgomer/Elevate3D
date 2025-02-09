import os
import numpy as np
import cv2
from scipy.ndimage import sobel, gaussian_gradient_magnitude
from scipy.interpolate import griddata
from src.configs.dsm2dtm_config import IMAGE_WIDTH, IMAGE_HEIGHT, HALF_WIDTH

def load_and_split_combined_image(file_path):
    """Load a combined DSM-DTM image and split it into two halves."""
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Error loading file: {file_path}")
    if img.shape != (IMAGE_HEIGHT, IMAGE_WIDTH):
        raise ValueError(f"Invalid dimensions: {img.shape}, expected ({IMAGE_HEIGHT}, {IMAGE_WIDTH})")
    
    dsm = img[:, :HALF_WIDTH].astype(np.float32)
    dtm = img[:, HALF_WIDTH:].astype(np.float32)
    return dsm, dtm

def interpolate_missing_data(dtm, nodata_value=0):
    """Interpolate missing values in DTM (assumes NoData is stored as '0')."""
    dtm[dtm == nodata_value] = np.nan
    coords = np.array(np.where(~np.isnan(dtm))).T
    values = dtm[~np.isnan(dtm)]
    grid_x, grid_y = np.indices(dtm.shape)
    interpolated_dtm = griddata(coords, values, (grid_x, grid_y), method="linear")
    return np.where(np.isnan(dtm), interpolated_dtm, dtm)

def extract_features(dsm):
    """Extract slope and gradient magnitude features from DSM."""
    slope_x = sobel(dsm, axis=0)
    slope_y = sobel(dsm, axis=1)
    slope_magnitude = np.sqrt(slope_x**2 + slope_y**2)
    gradient_magnitude = gaussian_gradient_magnitude(dsm, sigma=3)
    return np.stack([dsm, slope_magnitude, gradient_magnitude], axis=-1)
