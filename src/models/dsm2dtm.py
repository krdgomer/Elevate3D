import cv2
import numpy as np

def generate_dtm(dsm_path, output_path, kernel_size=10, iterations=2, smoothing_ksize=5):
    # Load DSM and convert to grayscale
    dsm = cv2.imread(dsm_path, cv2.IMREAD_UNCHANGED)
    dsm_gray = cv2.cvtColor(dsm, cv2.COLOR_BGR2GRAY) if len(dsm.shape) == 3 else dsm
    
    # Define morphological kernel
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    # Apply erosion to remove small features
    dtm = cv2.erode(dsm_gray, kernel, iterations=iterations)
    
    # Apply Gaussian smoothing to reduce pixelation
    dtm_smooth = cv2.GaussianBlur(dtm, (smoothing_ksize, smoothing_ksize), 0)
    
    # Save the result as PNG
    cv2.imwrite(output_path, dtm_smooth)
 
