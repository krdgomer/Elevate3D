import os
from huggingface_hub import hf_hub_download

class DownloadManager:
    def __init__(self, repo_id="krdgomer/elevate3d-weights", cache_dir="hf_cache"):
        """
        Initialize the DownloadManager with a Hugging Face repository ID and cache directory.

        Args:
            repo_id (str): The Hugging Face repository ID.
            cache_dir (str): The directory to cache downloaded files.
        """
        self.repo_id = repo_id
        self.cache_dir = cache_dir

    def download_file(self, filename, force_download=False):
        """
        Download a file from the Hugging Face Hub.

        Args:
            filename (str): The name of the file to download.
            force_download (bool): Whether to force re-download the file.

        Returns:
            str: The local path to the downloaded file.
        """
        try:
            print(f"Downloading {filename} from Hugging Face Hub...")
            file_path = hf_hub_download(
                repo_id=self.repo_id,
                filename=filename,
                cache_dir=self.cache_dir,
                force_download=force_download
            )
            print(f"Downloaded {filename} to {file_path}")
            return file_path
        except Exception as e:
            print(f"Error downloading {filename}: {e}")
            return None