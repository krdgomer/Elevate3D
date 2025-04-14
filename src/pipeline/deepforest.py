from deepforest import main
from deepforest import get_data
from deepforest.visualize import plot_results

def run_deepforest(image_path):

    print("Running DeepForest...")
    model = main.deepforest()

    model.load_model(model_name="weecology/deepforest-tree", revision="main")

    image_path = get_data(image_path)
    boxes = model.predict_image(path=image_path) 
    plot_results(boxes)
    return boxes