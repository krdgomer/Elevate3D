import os
import shutil
import random

# Define folder paths
train_folder = "src/datasets/rgb_dsm/train"
test_folder = "src/datasets/rgb_dsm/test"
val_folder = "src/datasets/rgb_dsm/validation"

# Ensure target directories exist
os.makedirs(test_folder, exist_ok=True)
os.makedirs(val_folder, exist_ok=True)

# Get all image files in train_folder
images = [f for f in os.listdir(train_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'))]

# Shuffle images for randomness
random.shuffle(images)

# Determine number of files to move
total_images = len(images)
num_b = int(total_images * 0.1)  # 10% for folder B
num_c = int(total_images * 0.1)  # 10% for folder C

# Move images to folder B
for img in images[:num_b]:
    shutil.move(os.path.join(train_folder, img), os.path.join(test_folder, img))

# Move images to folder C
for img in images[num_b:num_b + num_c]:
    shutil.move(os.path.join(train_folder, img), os.path.join(val_folder, img))

print(f"Moved {num_b} images to {test_folder}")
print(f"Moved {num_c} images to {val_folder}")
print(f"Remaining {len(os.listdir(train_folder))} images in {train_folder}")
