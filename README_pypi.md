# Elevate3D

**Generate 3D models from satellite & aerial images using deep learning.**

---

##  Disclaimer

**Elevate3D is an experimental, early-stage project.**  
It’s not production-ready, and results may be inconsistent depending on the input. Expect rough outputs, strange artifacts, and occasional surprises.


---

##  Features

-  Automatic Building Segmentation (Mask R-CNN)  
-  Elevation Prediction from RGB (Pix2Pix)  
-  3D Mesh Generation (Open3D)  
-  Tree Detection with DeepForest  
-  Pretrained Models – No training required  
-  End-to-End Pipeline – From image to interactive 3D output

---

##  Installation

Install with pip:

```bash
pip install elevate3d
```

---

##  Usage

### 1. Web Interface (Recommended)

Run the local web app:

```bash
elevate3d-run
```

This launches a local server where you can upload images and view results interactively in your browser.

### 2. Python API

You can also run the pipeline programmatically:

```python
from elevate3d.run_pipeline import run_pipeline

run_pipeline("path_to_your_image.jpg")
```

This processes the image and opens a viewer window showing the 3D model.

---

##  Input Requirements

- Accepts **aerial or satellite RGB images** (`.jpg`, `.jpeg`, `.png`)
- Images should ideally be top-down and contain visible buildings or tree cover

---

##  How It Works

- **Mask R-CNN** segments buildings
- **Pix2Pix** generates DSM from RGB
- **Tree detection** adds tree geometry
- **Open3D** constructs the final mesh (as `.glb`)

---

##  License

MIT License. See [`LICENSE`](LICENSE) for details.
