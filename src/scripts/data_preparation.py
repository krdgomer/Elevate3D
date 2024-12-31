import shutil
import os
from google.colab import drive

def prepare_data():
    # Mount drive
    drive.mount('/content/drive')
    
    # Create local directories
    os.makedirs('/content/local_train', exist_ok=True)
    os.makedirs('/content/local_val', exist_ok=True)
    
    # Copy training files locally
    source_train = "/content/drive/MyDrive/ProjeDosyalari/rgb_dsm_train/"
    for file in os.listdir(source_train):
        if file.endswith('.tif'):
            shutil.copy2(
                os.path.join(source_train, file),
                os.path.join('/content/local_train', file)
            )
    
    # Copy validation files locally
    source_val = "/content/drive/MyDrive/ProjeDosyalari/rgb_dsm_val/"
    for file in os.listdir(source_val):
        if file.endswith('.tif'):
            shutil.copy2(
                os.path.join(source_val, file),
                os.path.join('/content/local_val', file)
            )

    return '/content/local_train', '/content/local_val'