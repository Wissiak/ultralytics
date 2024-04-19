'''
This script converts a trained YOLO model to a TensorFlow.js model (use yolo env).
'''
from ultralytics import YOLO

# Load a model
path = "/Users/patrick/Downloads/epoch140.pt"
model = YOLO(f'{path}')  # load a custom trained model

# Export the model
# output saved to {path}/best_web_model
model.export(format='tfjs')