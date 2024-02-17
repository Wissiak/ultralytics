from ultralytics import YOLO

# Load a model
model = YOLO('/Users/patrick/projects/ultralytics_git/ultralytics/cfg/models/v8/yolov8n-corners.yaml')  # build a new model from YAML
#model = YOLO('yolov8n-obb.pt')  # load a pretrained model (recommended for training)
#model = YOLO('yolov8n-obb.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

# Train the model
results = model.train(data='/Users/patrick/projects/digicamp/dataset/data.yaml', epochs=100, imgsz=640, device="mps")