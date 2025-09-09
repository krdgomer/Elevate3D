from elevate3d.utils.download_manager import DownloadManager
import open3d as o3d
import numpy as np

class TextureManager:
    def __init__(self):
        self.download_manager = DownloadManager()

    def load_texture(self, texture_name):
        """Load texture from Hugging Face Hub"""
        try:
            texture_path = self.download_manager.download_file(texture_name)
            return o3d.io.read_image(texture_path)
        except Exception as e:
            print(f"Failed to load texture {texture_name}: {e}")
            return self._create_fallback_texture()
    
    def create_fallback_texture(self):
        """Create a simple fallback texture"""
        # Create a simple colored texture
        texture_array = np.ones((64, 64, 3), dtype=np.uint8) * 128
        return o3d.geometry.Image(texture_array)