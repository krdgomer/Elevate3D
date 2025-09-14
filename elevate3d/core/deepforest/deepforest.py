from deepforest import main
from deepforest import get_data
from deepforest.visualize import plot_results

def run_deepforest(image_path):
    """Runs the DeepForest model on the provided image. DeepForest is a python package for training and predicting ecological objects in airborne imagery.

    Args:
        image_path (string): Path to the image file.

    Returns:
        boxes: a list of bounding boxes predicted by the model.
    """
   
    model = main.deepforest()

    model.load_model(model_name="weecology/deepforest-tree", revision="main")

    image_path = get_data(image_path)
    boxes = model.predict_image(path=image_path) 
    return boxes