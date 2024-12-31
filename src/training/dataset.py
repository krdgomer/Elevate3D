import os
import random
import numpy as np
import rasterio
from torch.utils.data import Dataset
import torch
from src.configs import train_config as config


class TifDataset(Dataset):
    def __init__(self, root_dir, patch_size=512):
        self.root_dir = root_dir
        self.patch_size = patch_size

        # Tüm TIF dosyalarını bul
        all_files = os.listdir(root_dir)

        # DSM dosyalarını bul (tüm şehirler için)
        self.dsm_files = [f for f in all_files if 'DSM.tif' in f]

        # Her DSM dosyası için prefix ve tile numarasını ayır
        self.file_info = []
        for dsm_file in self.dsm_files:
            # Dosya adını parçalara ayır
            parts = dsm_file.split('_')
            prefix = parts[0]  # JAX, RIC veya TAM
            tile_num = parts[2]  # Tile numarası

            # RGB dosyasının varlığını kontrol et
            rgb_file = f"{prefix}_Tile_{tile_num}_RGB.tif"
            if rgb_file in all_files:
                self.file_info.append({
                    'prefix': prefix,
                    'tile_num': tile_num,
                    'dsm_file': dsm_file,
                    'rgb_file': rgb_file
                })

    def __len__(self):
        return len(self.file_info) * (2048 // self.patch_size) ** 2

    def __getitem__(self, index):
        # Hangi görüntü ve hangi patch
        img_idx = index // ((2048 // self.patch_size) ** 2)
        patch_idx = index % ((2048 // self.patch_size) ** 2)

        # Dosya bilgilerini al
        file_info = self.file_info[img_idx]

        # Dosya yollarını oluştur
        rgb_path = os.path.join(self.root_dir, file_info['rgb_file'])
        dsm_path = os.path.join(self.root_dir, file_info['dsm_file'])

        # Patch pozisyonunu hesapla
        row = (patch_idx // (2048 // self.patch_size)) * self.patch_size
        col = (patch_idx % (2048 // self.patch_size)) * self.patch_size

        try:
            # RGB görüntüsünü oku
            with rasterio.open(rgb_path) as src:
                rgb = src.read(
                    window=((row, row + self.patch_size),
                            (col, col + self.patch_size))
                )

            grayscale = 0.2989 * rgb[0] + 0.5870 * rgb[1] + 0.1140 * rgb[2]
            grayscale = grayscale.astype(np.float32)

            # DSM görüntüsünü oku
            with rasterio.open(dsm_path) as src:
                dsm = src.read(
                    1,  # İlk band
                    window=((row, row + self.patch_size),
                            (col, col + self.patch_size))
                )

            # Normalizasyon
            grayscale = grayscale / 255.0
            dsm = dsm.astype(np.float32)

            # Augmentation uygula
            augmentations = config.both_transform(image=grayscale, image0=dsm)
            grayscale = augmentations["image"]
            dsm = augmentations["image0"]

            grayscale = config.transform_only_input(image=grayscale)["image"]
            dsm = config.transform_only_mask(image=dsm)["image"]

            return grayscale, dsm

        except Exception as e:
            print(f"Error loading files: {rgb_path} or {dsm_path}")
            print(f"Error message: {str(e)}")
            # Hata durumunda başka bir görüntü dene
            return self.__getitem__(random.randint(0, len(self) - 1))

