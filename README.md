# Elevate3D

Generate 3D models from satellite & aerial images using deep learning

## Overview

Elevate3D is an early-stage, experimental Python tool that takes RGB images and generates basic 3D models using Mask R-CNN for building detection, Pix2Pix for elevation prediction, and Open3D for mesh generation.

⚠ This is NOT a professional-grade tool. It's in a very early state, and results may be highly inconsistent. Expect rough outputs, and feel free to experiment or modify the code to improve it.

## Features

✅ Automatic Building Segmentation (Mask R-CNN)\
✅ Elevation Prediction (Pix2Pix)\
✅ 3D Mesh Generation (Open3D)\
✅ Pretrained Weights Included – No training required\
✅ End-to-End Pipeline – Input images, output a 3D model

## Installation

1. Download the latest release (`.rar` file) from the [Releases](https://github.com/krdgomer/Elevate3D/releases) page.
2. Unzip the contents.
3. Install dependencies:
   
```
pip install -r requirements.txt
```

4. You're ready to generate 3D models!

## Usage

Run the following command to generate a 3D model from an input image:

```
python src/run_pipeline.py --image_path "path/to/your/image.png"
```
