import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

def load_and_split_image(img_path):
    """
    Load a combined image, split it into input and prediction, and convert to grayscale.
    """
    image = np.array(Image.open(img_path))

    # Split into input and prediction images
    input_image = image[:, :256, :]
    prediction_image = image[:, 256:, :]

    # Convert both images to grayscale
    input_image = np.array(Image.fromarray(input_image).convert("L"))
    prediction_image = np.array(Image.fromarray(prediction_image).convert("L"))

    return input_image, prediction_image

def calculate_errors(input_image, prediction_image):
    """
    Calculate the error (difference) between input and prediction images.
    Group errors into 16-bit chunks and compute MSE for each chunk.
    """
    # Calculate pixel-wise error
    error = input_image.astype(np.int16) - prediction_image.astype(np.int16)
    squared_error = error ** 2

    # Group errors into 16-bit chunks (0-16, 16-32, ..., 240-256)
    bins = np.arange(0, 257, 16)
    chunk_indices = np.digitize(input_image, bins) - 1

    # Calculate MSE for each chunk
    mse_per_chunk = []
    for i in range(len(bins) - 1):
        mask = (chunk_indices == i)
        if np.any(mask):
            mse = np.mean(squared_error[mask])
        else:
            mse = 0
        mse_per_chunk.append(mse)

    return error, squared_error, mse_per_chunk, bins

def visualize_error_distribution(mse_per_chunk, bins, save_path):
    """
    Generate a histogram of MSE per 16-bit chunk for the entire folder.
    """
    plt.figure()
    plt.bar(range(len(mse_per_chunk)), mse_per_chunk, tick_label=[f"{bins[i]}" for i in range(len(bins) - 1)])
    plt.xlabel("Pixel Value Chunk")
    plt.ylabel("Mean Squared Error (MSE)")
    plt.title("MSE per 16-bit Chunk ")
    plt.savefig(save_path)
    plt.close()

def create_error_visualization(input_image, error, img_name, save_dir):
    """
    Create a visualization of errors using a color palette (red for high error).
    Save the input image and error colormap together in a single image.
    """
    # Ensure all error values are positive
    error = np.abs(error)

    # Normalize error to [0, 1] for colormap
    error_normalized = (error - np.min(error)) / (np.max(error) - np.min(error))

    # Apply a red color map (higher error = more red)
    error_colormap = plt.get_cmap("Reds")(error_normalized)

    # Convert input image to RGB
    input_image_rgb = np.stack([input_image] * 3, axis=-1) / 255.0

    # Concatenate input image and error colormap horizontally
    combined_image = np.concatenate((input_image_rgb, error_colormap[:, :, :3]), axis=1)

    # Ensure the combined image values are in the range [0, 1]
    combined_image = np.clip(combined_image, 0, 1)

    # Save the combined image
    plt.imsave(os.path.join(save_dir, f"{img_name}_combined_visualization.png"), combined_image)

def count_error_distribution(error):
    """
    Count how many pixels fall into specific error ranges.
    """
    error_ranges = [0, 5, 10, 20, 30, 50, 100, 150, 255]
    counts = []
    for i in range(len(error_ranges) - 1):
        lower = error_ranges[i]
        upper = error_ranges[i + 1]
        count = np.sum((error >= lower) & (error < upper))
        counts.append(count)

    return error_ranges, counts

def process_images_in_folder(folder_path, save_dir):
    """
    Process all combined images in a folder.
    """
    os.makedirs(save_dir, exist_ok=True)
    image_files = [f for f in os.listdir(folder_path) if f.endswith((".png", ".jpg", ".jpeg"))]

    # Accumulate errors and squared errors for the entire folder
    all_errors = []
    all_squared_errors = []
    all_mse_per_chunk = []
    all_error_counts = []

    for idx, img_name in enumerate(tqdm(image_files)):
        img_path = os.path.join(folder_path, img_name)

        try:
            # Load and split the image
            input_image, prediction_image = load_and_split_image(img_path)

            # Calculate errors and MSE per chunk
            error, squared_error, mse_per_chunk, bins = calculate_errors(input_image, prediction_image)

            # Accumulate errors and squared errors
            all_errors.extend(error.flatten())
            all_squared_errors.extend(squared_error.flatten())

            # Accumulate MSE per chunk
            if not all_mse_per_chunk:
                all_mse_per_chunk = mse_per_chunk
            else:
                all_mse_per_chunk = [sum(x) for x in zip(all_mse_per_chunk, mse_per_chunk)]

            # Count error distribution
            error_ranges, error_counts = count_error_distribution(error)
            all_error_counts.append(error_counts)

            # Create error visualization for every 20th image
            if idx % 20 == 0:
                create_error_visualization(input_image, error, img_name, save_dir)

        except Exception as e:
            print(f"Error processing {img_name}: {e}")
            continue

    # Calculate MSE for the entire folder
    mse = np.mean(all_squared_errors)
    print(f"Mean Squared Error (MSE) for the entire folder: {mse:.4f}")

    # Calculate average MSE per chunk
    avg_mse_per_chunk = [mse / len(image_files) for mse in all_mse_per_chunk]

    # Generate histogram of MSE per chunk for the entire folder
    visualize_error_distribution(avg_mse_per_chunk, bins, os.path.join(save_dir, "mse_histogram.png"))

    plt.figure()
    plt.bar(range(len(error_counts)), error_counts, tick_label=[f"{error_ranges[i]}-{error_ranges[i + 1]}" for i in range(len(error_ranges) - 1)])
    plt.xlabel("Error Range")
    plt.ylabel("Pixel Count")
    plt.title("Error Distribution")
    plt.savefig(save_dir + "/error_distribution.png")
    plt.close()

    # Print error distribution for the entire folder
    total_error_counts = {}
    for counts in all_error_counts:
        for range_str, count in counts:
            if range_str in total_error_counts:
                total_error_counts[range_str] += count
            else:
                total_error_counts[range_str] = count

    print("Error distribution for the entire folder:")
    for range_str, count in total_error_counts.items():
        print(f"{range_str}: {count} pixels")

# Example usage
folder_path = "src/rgb2dsm/models/v0.2/predictions"
save_dir = "src/rgb2dsm/models/v0.2/test_results"
process_images_in_folder(folder_path, save_dir)