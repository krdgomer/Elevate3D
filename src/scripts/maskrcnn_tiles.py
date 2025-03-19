import os
from PIL import Image
import numpy as np

# Folder paths
rgb_input_folder = 'data/processed/maskrcnn/rgb'  # Folder containing RGB images
gti_input_folder = 'data/processed/maskrcnn/gti'  # Folder containing GTI images
rgb_output_folder = 'data/processed/maskrcnn/rgb_tiles'  # Folder to save RGB tiles
gti_output_folder = 'data/processed/maskrcnn/gti_tiles'  # Folder to save GTI tiles

# Create output folders if they don't exist
os.makedirs(rgb_output_folder, exist_ok=True)
os.makedirs(gti_output_folder, exist_ok=True)

# List all files in the RGB input folder
rgb_files = sorted([f for f in os.listdir(rgb_input_folder) if f.endswith('.png')])
gti_files = sorted([f for f in os.listdir(gti_input_folder) if f.endswith('.png')])

# Ensure the files are paired correctly
assert len(rgb_files) == len(gti_files), "Mismatch between RGB and GTI files!"

# Process each pair of RGB and GTI files
for rgb_file, gti_file in zip(rgb_files, gti_files):
    # Open the RGB image
    rgb_path = os.path.join(rgb_input_folder, rgb_file)
    rgb_image = Image.open(rgb_path).convert('RGB')  # Ensure RGB format

    # Open the GTI image
    gti_path = os.path.join(gti_input_folder, gti_file)
    gti_image = Image.open(gti_path)

    # Check if both images are 2048x2048
    if rgb_image.size == (2048, 2048) and gti_image.size == (2048, 2048):
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

                # Crop the RGB tile
                rgb_tile = rgb_image.crop((left, upper, right, lower))

                # Convert the RGB tile to a numpy array
                rgb_tile_array = np.array(rgb_tile)

                # Check if the RGB tile contains any blank (black) pixels
                if np.any(np.all(rgb_tile_array == 0, axis=-1)):  # Check for pure black pixels
                    print(f"Skipping tile {i * 4 + j + 1} in {rgb_file}: RGB tile contains blank pixels")
                    continue  # Skip this tile

                # Crop the GTI tile
                gti_tile = gti_image.crop((left, upper, right, lower))

                # Convert the GTI tile to a numpy array
                gti_tile_array = np.array(gti_tile)

                # Check if the GTI tile contains any buildings (non-zero pixels)
                if np.all(gti_tile_array == 0):  # Check if all pixels are 0 (no buildings)
                    print(f"Skipping tile {i * 4 + j + 1} in {gti_file}: GTI tile contains no buildings")
                    continue  # Skip this tile

                # Create the new filenames
                base_name = rgb_file.replace('_RGB.png', '')  # Remove suffix
                tile_id = i * 4 + j + 1  # Tile number (1 to 16)
                rgb_tile_filename = f"{base_name}_RGB_tile_{tile_id}.png"
                gti_tile_filename = f"{base_name}_GTI_tile_{tile_id}.png"

                # Save the RGB tile
                rgb_tile.save(os.path.join(rgb_output_folder, rgb_tile_filename))

                # Save the GTI tile
                gti_tile.save(os.path.join(gti_output_folder, gti_tile_filename))

                print(f"Saved: {rgb_tile_filename} and {gti_tile_filename}")
    else:
        print(f"Skipping {rgb_file} and {gti_file}: Not 2048x2048 images")