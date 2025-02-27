from PIL import Image
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

class ErrorCalculator():
    def __init__(self, predictions_path,version):
        self.predictions_path = predictions_path
        self.all_errors = []
        self.all_squared_errors = []
        self.all_mse_per_chunk = []
        self.all_error_counts = []
        self.version = version

    def load_and_split_image(self,img_path):
        image = np.array(Image.open(img_path))

        input_image = image[:, :512, :]
        prediction_image = image[:, 512:, :]

        input_image = np.array(Image.fromarray(input_image).convert("L"))
        prediction_image = np.array(Image.fromarray(prediction_image).convert("L"))

        return input_image, prediction_image

    def count_error_distribution(self,error):
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

    def visualize_error_distribution(self,mse_per_chunk, bins, save_path):
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

    def create_error_visualization(self,input_image, error, img_name, save_dir):
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

    def log_results(self, save_dir, mse, avg_mse_per_chunk, total_error_counts):
        """
        Save the calculated metrics and error distribution results into a text file.
        """
        results_file_path = os.path.join(save_dir, "results.txt")
        with open(results_file_path, "w") as f:
            f.write(f"Results of Error Calculation ({self.version})\n")
            f.write("============================\n\n")
            f.write(f"Mean Squared Error (MSE) for the entire folder: {mse:.4f}\n\n")
            f.write("Average MSE per 16-bit chunk:\n")
            for i, mse_chunk in enumerate(avg_mse_per_chunk):
                f.write(f"Chunk {i * 16}-{(i + 1) * 16}: {mse_chunk:.4f}\n")
            f.write("\nError Distribution for the entire folder:\n")
            for range_str, count in total_error_counts.items():
                f.write(f"{range_str}: {count} pixels\n")

    def calculate_errors(self,save_dir):
        metrics = {
                "mse": [],
                "rmse": [],
                "mae": []
            }

        image_files = [f for f in os.listdir(self.predictions_path) if f.endswith((".png", ".jpg", ".jpeg"))]

        for idx,img_name in enumerate(tqdm(image_files)):
            img_path = os.path.join(self.predictions_path, img_name)

            try:
                
                input_image, prediction_image = self.load_and_split_image(img_path)

                error = input_image.astype(np.int16) - prediction_image.astype(np.int16)
                squared_error = error ** 2

                bins = np.arange(0, 257, 16)
                chunk_indices = np.digitize(input_image, bins) - 1

                mse_per_chunk = []
                for i in range(len(bins) - 1):
                    mask = (chunk_indices == i)
                    if np.any(mask):
                        mse = np.mean(squared_error[mask])
                    else:
                        mse = 0
                    mse_per_chunk.append(mse)

                self.all_errors.extend(error.flatten())
                self.all_squared_errors.extend(squared_error.flatten())

                           
                if not self.all_mse_per_chunk:
                    self.all_mse_per_chunk = mse_per_chunk
                else:
                    self.all_mse_per_chunk = [sum(x) for x in zip(self.all_mse_per_chunk, mse_per_chunk)]

                error_ranges, error_counts = self.count_error_distribution(error)
                self.all_error_counts.append(error_counts)

                """if idx % 20 == 0:
                    self.create_error_visualization(input_image, error, img_name, save_dir)"""
            
            except Exception as e:
                print(f"Error processing {img_name}: {e}")
            continue

            
        mse = np.mean(self.all_squared_errors)
        print(f"Mean Squared Error (MSE) for the entire folder: {mse:.4f}")
       
        avg_mse_per_chunk = [mse / len(image_files) for mse in self.all_mse_per_chunk]

        self.visualize_error_distribution(avg_mse_per_chunk, bins, os.path.join(save_dir, "mse_histogram.png"))

        # Initialize total_error_counts with zeros for all ranges
        total_error_counts = {f"{error_ranges[i]}-{error_ranges[i + 1]}": 0 for i in range(len(error_ranges) - 1)}

        # Accumulate error counts across all images
        for counts in self.all_error_counts:
            for i in range(len(error_ranges) - 1):
                range_str = f"{error_ranges[i]}-{error_ranges[i + 1]}"
                total_error_counts[range_str] += counts[i]

        # Plot the accumulated error counts
        plt.figure()
        plt.bar(range(len(total_error_counts)), list(total_error_counts.values()), tick_label=list(total_error_counts.keys()))
        plt.xlabel("Error Range")
        plt.ylabel("Pixel Count")
        plt.title("Error Distribution")
        plt.savefig(os.path.join(save_dir, "error_distribution.png"))
        plt.close()

        # Print the accumulated error counts
        print("Error distribution for the entire folder:")
        for range_str, count in total_error_counts.items():
            print(f"{range_str}: {count} pixels")

        self.log_results(save_dir, mse, avg_mse_per_chunk, total_error_counts)