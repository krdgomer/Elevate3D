from PIL import Image
import os
import numpy as np

# Folder paths
input_folder = 'data/processed/maskrcnn/gti'  # Path where the original images are stored
output_folder = 'data/processed/maskrcnn/gti_tiles'  # Path where the split tiles will be saved

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# List all files in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith('_GTI.png'): 
        # Open the image
        img_path = os.path.join(input_folder, filename)
        img = Image.open(img_path)
        
        # Check if the image is 2048x2048
        if img.size == (2048, 2048):
            # Define tile size (512x512)
            tile_size = 512
            
            # Loop over the 16 tiles (4x4 grid)
            for i in range(4):
                for j in range(4):
                    # Calculate the box to crop the image
                    left = j * tile_size
                    upper = i * tile_size
                    right = left + tile_size
                    lower = upper + tile_size
                    
                    # Crop the image
                    tile = img.crop((left, upper, right, lower))
                    
                    # Convert the tile to a numpy array
                    tile_array = np.array(tile)
                    """
                    # Check if the tile contains any blank (black) pixels
                    if np.any(tile_array == 0):  # Assuming black pixels have value 0
                        print(f"Skipping tile {i * 4 + j + 1} in {filename}: Contains blank pixels")
                        continue  # Skip this tile"""
                    
                    # Create the new filename
                    new_filename = f"gti_{filename[4:-4]}_{i * 4 + j + 1}.png"
                    new_file_path = os.path.join(output_folder, new_filename)
                    
                    # Save the cropped tile
                    tile.save(new_file_path)

                    print(f"Saved: {new_filename}")
        else:
            print(f"Skipping {filename}: Not a 2048x2048 image")