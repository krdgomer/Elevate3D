from PIL import Image
import os

def split_and_save_images(input_folder, output_folder, tile_size=512):
    """
    Splits images in the input folder into tiles of tile_size x tile_size dimensions, ignoring
    incomplete tiles, and saves them in the output folder.

    Args:
        input_folder (str): Path to the folder containing input images.
        output_folder (str): Path to the folder where output tiles will be saved.
        tile_size (int): The size of each square tile (default: 512).
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate through all image files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".png"):
            img_path = os.path.join(input_folder, filename)
            img = Image.open(img_path)
            width, height = img.size

            base_name = os.path.splitext(filename)[0]
            count = 1

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
                    tile_filename = f"{base_name}_{count}.png"
                    tile.save(os.path.join(output_folder, tile_filename))
                    count += 1

            print(f"Processed {filename}, generated {count - 1} complete tiles.")

# Define input and output folders
input_folder = "data/processed"
output_folder = "src/datasets/telaviv"

# Call the function
split_and_save_images(input_folder, output_folder)
