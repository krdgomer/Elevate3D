import os
import rasterio
from PIL import Image
import numpy as np

# Define the input and output folders
rgb_input_folder = "data/raw/rgb_tif"
gti_input_folder = "data/raw/gti_tif"
rgb_output_folder = "data/processed/maskrcnn/rgb"
gti_output_folder = "data/processed/maskrcnn/gti"

# Create the output folders if they don't exist
os.makedirs(rgb_output_folder, exist_ok=True)
os.makedirs(gti_output_folder, exist_ok=True)

# Get a list of all files in the input folders
rgb_files = sorted([f for f in os.listdir(rgb_input_folder) if f.endswith("_RGB.tif")])
gti_files = sorted([f for f in os.listdir(gti_input_folder) if f.endswith("_GTI.tif")])

# Ensure the files are paired correctly
assert len(rgb_files) == len(gti_files), "Mismatch between RGB and GTI files!"

# Process each pair of RGB and GTI files
for rgb_file, gti_file in zip(rgb_files, gti_files):
    # Extract the base title_id
    title_id = rgb_file.replace("_RGB.tif", "")

    # Read the RGB image
    rgb_path = os.path.join(rgb_input_folder, rgb_file)
    with rasterio.open(rgb_path) as src:
        rgb_image = src.read()  # Read all bands (3 bands for RGB)
        rgb_image = np.moveaxis(rgb_image, 0, -1)  # Move bands to the last dimension (HxWxC)

    # Read the GTI mask
    gti_path = os.path.join(gti_input_folder, gti_file)
    with rasterio.open(gti_path) as src:
        gti_mask = src.read(1)  # Read the first band (16-bit mask)

    # Save the RGB image as PNG
    rgb_output_path = os.path.join(rgb_output_folder, f"{title_id}_RGB.png")
    rgb_image_pil = Image.fromarray(rgb_image.astype(np.uint8))  # Convert to 8-bit for RGB
    rgb_image_pil.save(rgb_output_path, format="PNG")

    # Save the GTI mask as 16-bit PNG
    gti_output_path = os.path.join(gti_output_folder, f"{title_id}_GTI.png")
    gti_mask_pil = Image.fromarray(gti_mask.astype(np.uint16))  # Convert to 16-bit for mask
    gti_mask_pil.save(gti_output_path, format="PNG", bits=16)

    print(f"Processed: {title_id}")

print("Conversion complete!")