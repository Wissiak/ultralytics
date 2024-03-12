from ultralytics import YOLO

# Load a model
model = YOLO('/home/patrick.wissiak/work/ultralytics/ultralytics/cfg/models/v8/yolov8n.yaml')  # build a new model from YAML
#model = YOLO('yolov8n-obb.pt')  # load a pretrained model (recommended for training)
#model = YOLO('yolov8n-obb.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

# Train the model
results = model.train(
    task='detect',
    mode='train',
    data='/home/patrick.wissiak/work/ultralytics/ultralytics/cfg/datasets/coco8.yaml', 
    #data='/Users/patrick/projects/digicamp/dataset/data.yaml', 
    epochs=100,
    imgsz=640,
    device="3"
)
# results = model.train(
#     data='/Users/patrick/projects/digicamp/dataset/data.yaml', 
#     epochs=100, 
#     imgsz=640,
#     device="mps",
#     augment=False
# )