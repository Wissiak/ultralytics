from ultralytics import YOLO

# Load a model
model = YOLO('/Users/patrick/projects/ultralytics_git/ultralytics/cfg/models/v8/yolov8s-corners.yaml')  # build a new model from YAML
#model = YOLO('yolov8n-obb.pt')  # load a pretrained model (recommended for training)
#model = YOLO('yolov8n-obb.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

# Train the model
results = model.train(
    data='/Users/patrick/projects/digicamp/dataset/data.yaml', 
    #weights='yolov8n.pt',
    epochs=100, 
    patience=15, 
    imgsz=640, 
    device="mps",
    task='detect',
    mode='train',
    augment=False,
    batch=16, 
    save_period=1, 
    deterministic=False,
    #dropout=0.2, 
    project='/Users/patrick/projects/ultralytics_git/runs', 
    verbose=True, 
)

# save_dir = /opt/homebrew/runs/corners/train310