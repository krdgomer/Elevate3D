import numpy as np
import cv2
import numba

def normalize_safe(array):
    """Safe normalization that handles constant arrays."""
    min_val, max_val = np.min(array), np.max(array)
    if max_val - min_val < 1e-10:  # Array is constant
        return np.zeros_like(array, dtype=np.float32)
    return (array - min_val) / (max_val - min_val)

def dsm_uint8_to_float(dsm_uint8, height_range=(0, 100)):
    """
    Convert DSM from uint8 [0, 255] to float32 with specified height range.
    
    Args:
        dsm_uint8: Input DSM as uint8 array
        height_range: Tuple of (min_height, max_height) in meters
    """
    dsm_normalized = dsm_uint8.astype(np.float32) / 255.0
    min_height, max_height = height_range
    dsm_float = dsm_normalized * (max_height - min_height) + min_height
    return dsm_float

def float_to_dsm_uint8(dsm_float):
    """Convert float DSM back to uint8."""
    dsm_normalized = normalize_safe(dsm_float)
    return (dsm_normalized * 255).astype(np.uint8)

@numba.jit(nopython=True)
def csf_simulation_iteration(cloth_heights, terrain_heights, rigidness=3.0, 
                           time_step=0.02, gravity=9.8, max_displacement=0.5):
    """
    Single iteration of cloth simulation.
    """
    height, width = cloth_heights.shape
    new_heights = cloth_heights.copy()
    
    for i in range(1, height-1):
        for j in range(1, width-1):
            # Current cloth height
            current_height = cloth_heights[i, j]
            
            # Terrain height at this position
            terrain_height = terrain_heights[i, j]
            
            # If cloth is above terrain, apply gravity
            if current_height > terrain_height:
                # Simple gravity effect
                velocity = gravity * time_step
                new_height = current_height - velocity
                
                # Constraint: cloth cannot go below terrain
                new_height = max(new_height, terrain_height)
                
                # Add some rigidity from neighbors
                neighbor_avg = (cloth_heights[i-1, j] + cloth_heights[i+1, j] + 
                               cloth_heights[i, j-1] + cloth_heights[i, j+1]) / 4.0
                
                # Blend between gravity effect and neighbor influence
                rigidness_normalized = min(max(rigidness / 10.0, 0.0), 1.0)
                new_heights[i, j] = (1.0 - rigidness_normalized) * new_height + rigidness_normalized * neighbor_avg
                
                # Limit maximum displacement
                displacement = abs(new_heights[i, j] - current_height)
                if displacement > max_displacement:
                    new_heights[i, j] = current_height + np.sign(new_heights[i, j] - current_height) * max_displacement
    
    return new_heights

def generate_dtm_csf(dsm_uint8, max_iterations=100, rigidness=2.0, 
                    time_step=0.02, gravity=15.0, convergence_threshold=0.01,
                    height_range=(0, 100), verbose=True):
    """
    Generate DTM using Cloth Simulation Filter (CSF).
    
    Args:
        dsm_uint8: Input DSM as uint8 image
        max_iterations: Maximum number of simulation iterations
        rigidness: How rigid the cloth is (1.0-5.0)
        time_step: Simulation time step
        gravity: Gravity strength
        convergence_threshold: Stop when changes are below this threshold
        height_range: Tuple of (min_height, max_height) for DSM scaling
        verbose: Print iteration progress
    
    Returns:
        dtm_uint8: DTM as uint8 image
        building_height_float: Building height as float array in meters
    """
    # Convert to float with meter scale
    dsm_float = dsm_uint8_to_float(dsm_uint8, height_range)
    height, width = dsm_float.shape
    
    # Initialize cloth HIGH ABOVE the surface - this was the key issue!
    max_height = np.max(dsm_float) + 50.0  # Much higher buffer for proper settling
    cloth_heights = np.full_like(dsm_float, max_height)
    
    # Better boundary conditions - don't constrain to DSM values!
    # Let boundaries settle naturally or use minimum terrain height
    min_terrain_estimate = np.percentile(dsm_float, 10)  # Use 10th percentile as ground estimate
    
    cloth_heights[0, :] = max_height * 0.8   # Let boundaries fall naturally
    cloth_heights[-1, :] = max_height * 0.8
    cloth_heights[:, 0] = max_height * 0.8
    cloth_heights[:, -1] = max_height * 0.8
    
    # Main simulation loop
    prev_heights = cloth_heights.copy()
    converged = False
    
    for iteration in range(max_iterations):
        cloth_heights = csf_simulation_iteration(
            cloth_heights, dsm_float, rigidness, time_step, gravity, 0.5
        )
        
        # Check for convergence
        max_change = np.max(np.abs(cloth_heights - prev_heights))
        if max_change < convergence_threshold:
            if verbose:
                print(f"CSF converged after {iteration + 1} iterations")
            converged = True
            break
        
        prev_heights = cloth_heights.copy()
        
        if verbose and iteration % 20 == 0:
            print(f"Iteration {iteration + 1}, max change: {max_change:.4f}")
    
    if not converged and verbose:
        print(f"CSF did not converge after {max_iterations} iterations")
    
    # The final cloth heights represent the DTM
    dtm_float = cloth_heights
    
    # Ensure DTM doesn't exceed DSM
    dtm_float = np.minimum(dtm_float, dsm_float)
    
    # Calculate building height
    building_height_float = dsm_float - dtm_float
    building_height_float = np.maximum(building_height_float, 0)
    
    # Convert back to uint8
    dtm_uint8 = float_to_dsm_uint8(dtm_float)
    
    return dtm_uint8, building_height_float

def generate_dtm_csf_optimized(dsm_uint8, max_iterations=50, rigidness=2.5,
                              height_range=(0, 100), scale_factor=0.5):
    """
    Optimized version of CSF using multi-scale approach.
    
    Args:
        dsm_uint8: Input DSM as uint8 array
        max_iterations: Maximum iterations for CSF
        rigidness: Cloth rigidness parameter
        height_range: Height range for scaling
        scale_factor: Downsampling factor for initial processing
    """
    # Convert to float
    dsm_float = dsm_uint8_to_float(dsm_uint8, height_range)
    
    # Step 1: Downsample for faster processing
    dsm_small = cv2.resize(dsm_float, None, fx=scale_factor, fy=scale_factor, 
                          interpolation=cv2.INTER_AREA)
    
    # Step 2: Run CSF on downsampled version
    dtm_small_uint8, _ = generate_dtm_csf(
        float_to_dsm_uint8(dsm_small), 
        max_iterations=max_iterations, 
        rigidness=rigidness,
        time_step=0.03,
        gravity=12.0,
        height_range=height_range,
        verbose=False
    )
    
    # Convert back to float and upsample
    dtm_small_float = dsm_uint8_to_float(dtm_small_uint8, height_range)
    dtm_float = cv2.resize(dtm_small_float, dsm_float.shape[::-1], 
                          interpolation=cv2.INTER_CUBIC)
    
    # Step 3: Refine with morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dtm_float = cv2.morphologyEx(dtm_float, cv2.MORPH_CLOSE, kernel)
    dtm_float = cv2.GaussianBlur(dtm_float, (7, 7), 2.0)
    
    # Ensure DTM is always below or equal to DSM
    dtm_float = np.minimum(dtm_float, dsm_float)
    
    # Calculate building height
    building_height_float = dsm_float - dtm_float
    building_height_float = np.maximum(building_height_float, 0)
    
    # Convert to uint8
    dtm_uint8 = float_to_dsm_uint8(dtm_float)
    
    return dtm_uint8, building_height_float

# Example usage with better error handling
def generate_dtm(dsm_uint8):
    """Example usage with comprehensive error handling and analysis."""
    try:
        
        # Generate DTM using optimized CSF
        print("Running Optimized Cloth Simulation Filter...")
        dtm_csf, height_csf = generate_dtm_csf_optimized(
            dsm_uint8, 
            max_iterations=50, 
            rigidness=2.5,
            height_range=(0, 100)  # Adjust based on your data
        )
        
        # Convert for analysis
        dsm_float = dsm_uint8_to_float(dsm_uint8, (0, 100))
        dtm_float = dsm_uint8_to_float(dtm_csf, (0, 100))
        
        dtm_float = dtm_float-dsm_float  
        dtm_csf = dtm_csf - dsm_uint8 
        
        return dtm_csf
        
    except Exception as e:
        print(f"Error in example_usage: {e}")
        return None, None
    

def generate_dtm3(dsm):
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
    




