import numpy as np

def generate_dtm(dsm):
    """
    Generate a blank DTM from an in-memory DSM (OpenCV format).
    
    Args:
        dsm_cv2: DSM image as OpenCV format (NumPy array, uint8, shape HxW)
        
    Returns:
        dtm_cv2: DTM image as OpenCV format (NumPy array, uint8, same shape as DSM)
    """
    # Create a black image with same dimensions as DSM
    dtm_cv2 = np.zeros_like(dsm, dtype=np.uint8)
    
    return dtm_cv2

