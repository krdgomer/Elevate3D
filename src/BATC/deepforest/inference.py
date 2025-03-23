from deepforest import main
from deepforest import get_data
from deepforest.visualize import plot_results
# Initialize the model class
model = main.deepforest()

# Load a pretrained tree detection model from Hugging Face
model.load_model(model_name="weecology/deepforest-tree", revision="main")

sample_image_path = get_data("D:/nyquil/dev/Bitirme-Projesi/Elevate3D/src/BATC/deepforest/image.png")
boxes = model.predict_image(path=sample_image_path,return_plot=False) 
print(boxes)