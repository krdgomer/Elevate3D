import numpy
import json
import pandas as pd
import os
import matplotlib.pyplot as plt
from skimage.io import imread
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import numpy as np
map_base_dir = 'src/BATC/datasets'
map_img_dir = 'src/BATC/datasets/train/images/'

json_path = os.path.join(map_base_dir, 'annotation.json')
with open(json_path, 'r') as f:
    annot_data = json.load(f)

annot_df = pd.DataFrame(annot_data['annotations'])
print(annot_df.sample(3))