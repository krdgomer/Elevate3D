from PIL import Image
import os

# Folder path where both dsm and dtm images are stored
folder_path = 'data/processed/dsm_dtm'  # Path where both dsm and dtm images are stored
output_folder = 'src/datasets/dsm_dtm_train'  # Path to save combined images

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# List all files in the folder
all_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]

# Split files into dsm and dtm lists based on filenames
dsm_files = [f for f in all_files if f.startswith('dsm_')]
dtm_files = [f for f in all_files if f.startswith('dtm_')]

# Extract the base numbers from filenames
dsm_bases = {f.split('_')[1] + '_' + f.split('_')[2][:-4]: f for f in dsm_files}
dtm_bases = {f.split('_')[1] + '_' + f.split('_')[2][:-4]: f for f in dtm_files}

# Loop through each matching base number
for base_number in dsm_bases.keys():
    if base_number in dtm_bases:
        # Open the dsm and dtm images
        dsm_img = Image.open(os.path.join(folder_path, dsm_bases[base_number]))
        dtm_img = Image.open(os.path.join(folder_path, dtm_bases[base_number]))

        # Check if both images have the same height
        if dsm_img.height == dtm_img.height:
            # Combine the images by pasting them side by side
            combined_img = Image.new('RGB', (dsm_img.width + dtm_img.width, dsm_img.height))
            combined_img.paste(dsm_img, (0, 0))  # dsm on the left
            combined_img.paste(dtm_img, (dsm_img.width, 0))  # dtm on the right

            # Create the combined filename
            combined_filename = f"combined_{base_number}.png"
            combined_file_path = os.path.join(output_folder, combined_filename)

            # Save the combined image
            combined_img.save(combined_file_path)

            print(f"Saved: {combined_filename}")
        else:
            print(f"Skipping {base_number}: Images have different heights")
    else:
        print(f"Skipping {base_number}: No matching dtm file")
