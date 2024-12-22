from PIL import Image
import os

def combine_dsm_rgb(input_folder, output_folder):
    """
    Combines DSM and RGB image pairs into a single image with RGB on the left and DSM on the right.

    Args:
        input_folder (str): Path to the folder containing both DSM and RGB images.
        output_folder (str): Path to the folder where combined images will be saved.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get list of DSM and RGB files
    dsm_files = sorted([f for f in os.listdir(input_folder) if f.endswith("_dsm.png")])
    rgb_files = sorted([f for f in os.listdir(input_folder) if f.endswith("_rgb.png")])

    combined_index = 1

    for dsm_file, rgb_file in zip(dsm_files, rgb_files):
        # Extract numbers from filenames to ensure they match
        dsm_number = os.path.splitext(dsm_file)[0].split("_")[0]
        rgb_number = os.path.splitext(rgb_file)[0].split("_")[0]

        if dsm_number != rgb_number:
            print(f"Skipping unmatched pair: {dsm_file}, {rgb_file}")
            continue

        # Open images
        dsm_path = os.path.join(input_folder, dsm_file)
        rgb_path = os.path.join(input_folder, rgb_file)
        dsm_img = Image.open(dsm_path)
        rgb_img = Image.open(rgb_path)

        # Verify image dimensions
        if dsm_img.size != (512, 512) or rgb_img.size != (512, 512):
            print(f"Skipping invalid image dimensions for pair: {dsm_file}, {rgb_file}")
            continue

        # Create a new image with combined dimensions
        combined_img = Image.new("RGB", (1024, 512))

        # Paste the RGB on the left and DSM on the right
        combined_img.paste(rgb_img, (0, 0))
        combined_img.paste(dsm_img, (512, 0))

        # Save the combined image
        combined_filename = f"{combined_index}.png"
        combined_img.save(os.path.join(output_folder, combined_filename))
        
        print(f"Saved combined image: {combined_filename}")
        combined_index += 1

# Define input and output folders
input_folder = "data/processed"
output_folder = "src/datasets/rgb_dsm"

# Call the function
combine_dsm_rgb(input_folder, output_folder)
