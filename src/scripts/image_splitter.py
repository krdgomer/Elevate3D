from PIL import Image
import os
import re

def split_and_save_images(input_folder, output_folder, tile_size=512):
    """
    Splits images in the input folder into tiles of tile_size x tile_size dimensions, ensuring
    that only complete tiles are saved, and names them based on their class and index.

    Args:
        input_folder (str): Path to the folder containing input images.
        output_folder (str): Path to the folder where output tiles will be saved.
        tile_size (int): The size of each square tile (default: 512).
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Group files by their prefix number and sort numerically
    files = [f for f in os.listdir(input_folder) if f.endswith(".png")]
    files.sort(key=lambda f: int(re.match(r"(\d+)_", f).group(1)))

    grouped_files = {}

    for file in files:
        match = re.match(r"(\d+)_([a-zA-Z]+)\.png", file)
        if match:
            prefix, class_name = match.groups()
            if prefix not in grouped_files:
                grouped_files[prefix] = {}
            grouped_files[prefix][class_name] = os.path.join(input_folder, file)

    # Initialize class-specific indices
    indices = {
        "dsm": 1,
        "rgb": 1
    }

    # Process each group of images
    for prefix, class_files in grouped_files.items():
        for class_name, file_path in class_files.items():
            if class_name not in indices:
                print(f"Skipping unknown class: {class_name}")
                continue

            img = Image.open(file_path)
            width, height = img.size

            # Calculate the number of complete tiles in both dimensions
            tiles_x = width // tile_size
            tiles_y = height // tile_size

            # Loop to extract only complete tiles
            for y in range(tiles_y):
                for x in range(tiles_x):
                    # Define the box for the current tile
                    left = x * tile_size
                    top = y * tile_size
                    right = left + tile_size
                    bottom = top + tile_size
                    box = (left, top, right, bottom)

                    # Crop and save the tile
                    tile = img.crop(box)
                    tile_filename = f"{indices[class_name]}_{class_name}.png"
                    tile.save(os.path.join(output_folder, tile_filename))

                    indices[class_name] += 1

            print(f"Processed {file_path}, generated tiles up to index {indices[class_name] - 1} for {class_name}.")

# Define input and output folders
input_folder = "data/raw"
output_folder = "data/processed"

# Call the function
split_and_save_images(input_folder, output_folder)
