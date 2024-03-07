from ultralytics import YOLO

# Load a model
model = YOLO('/home/patrick.wissiak/work/ultralytics/ultralytics/cfg/models/v8/yolov8-seg.yaml')  # build a new model from YAML
#model = YOLO('yolov8n-obb.pt')  # load a pretrained model (recommended for training)
#model = YOLO('yolov8n-obb.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

# Train the model
results = model.train(
    task='detect',
    mode='train',
    data='/home/patrick.wissiak/work/dataset/data.yaml', 
    #data='/Users/patrick/projects/digicamp/dataset/data.yaml', 
    save_period=5,
    epochs=150,
    patience=15,
    imgsz=640,
    device="0,1,2,3",
    augment=False,
    batch=64, 
    #deterministic=False,
    dropout=0.2, 
    project='/home/patrick.wissiak/work/runs', 
    verbose=True, 
)
# results = model.train(
#     data='/Users/patrick/projects/digicamp/dataset/data.yaml', 
#     epochs=100, 
#     imgsz=640,
#     device="mps",
#     augment=False
# )