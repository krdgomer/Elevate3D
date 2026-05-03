# Configuration for Elevate3D

# Image settings
IMAGE_SIZE = 512  # Required image size (width and height in pixels)

# Directory settings (relative to project root)
DATA_DIR = "data"  # Local data directory for uploads, models, etc.
UPLOAD_DIR = "uploads"  # Subdir for uploaded files
MODEL_DIR = "models"  # Subdir for generated models
CACHE_DIR = "hf_cache"  # Cache for downloaded weights

# Processing settings
HEIGHT_SCALE = 1.0  # Scale factor for building heights

# Logging settings
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR

# File upload settings
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB limit