from PIL import Image
import os

# Folder path where both rgb and dsm images are stored
folder_path = 'data/processed/rgb_dsm'  # Path where both rgb and dsm images are stored
output_folder = 'src/datasets/rgb_dsm_train'  # Path to save combined images

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# List all files in the folder
all_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]

# Split files into rgb and dsm lists based on filenames
rgb_files = [f for f in all_files if f.startswith('rgb_')]
dsm_files = [f for f in all_files if f.startswith('dsm_')]

# Extract the base numbers from filenames
rgb_bases = {f.split('_')[1] + '_' + f.split('_')[2][:-4]: f for f in rgb_files}
dsm_bases = {f.split('_')[1] + '_' + f.split('_')[2][:-4]: f for f in dsm_files}

# Loop through each matching base number
for base_number in rgb_bases.keys():
    if base_number in dsm_bases:
        # Open the RGB and DSM images
        rgb_img = Image.open(os.path.join(folder_path, rgb_bases[base_number]))
        dsm_img = Image.open(os.path.join(folder_path, dsm_bases[base_number]))

        # Check if both images have the same height
        if rgb_img.height == dsm_img.height:
            # Combine the images by pasting them side by side
            combined_img = Image.new('RGB', (rgb_img.width + dsm_img.width, rgb_img.height))
            combined_img.paste(rgb_img, (0, 0))  # RGB on the left
            combined_img.paste(dsm_img, (rgb_img.width, 0))  # DSM on the right

            # Create the combined filename
            combined_filename = f"combined_{base_number}.png"
            combined_file_path = os.path.join(output_folder, combined_filename)

            # Save the combined image
            combined_img.save(combined_file_path)

            print(f"Saved: {combined_filename}")
        else:
            print(f"Skipping {base_number}: Images have different heights")
    else:
        print(f"Skipping {base_number}: No matching DSM file")
