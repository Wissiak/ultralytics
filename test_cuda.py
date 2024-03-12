from ultralytics import YOLO

# Load a model
model = YOLO('/home/patrick.wissiak/work/ultralytics/ultralytics/cfg/models/v8/yolov8m-corners.yaml')  # build a new model from YAML
#model = YOLO('/home/patrick.wissiak/work/runs/train/weights/best.pt')  # build a new model from YAML
#model = YOLO('yolov8n-obb.pt')  # load a pretrained model (recommended for training)
#model = YOLO('yolov8n-obb.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

# Train the model
results = model.train(
    data='/home/patrick.wissiak/work/dataset/data.yaml', 
    epochs=250, 
    patience=15, 
    imgsz=640, 
    device="0,1,2,3",
    #device="0",
    task='detect',
    mode='train',
    augment=False,
    batch=64, 
    #save_period=20, 
    #deterministic=False,
    #dropout=0.2, 
    project='/home/patrick.wissiak/work/runs', 
    verbose=True, 
    #workers=0
)

# save_dir = /opt/homebrew/runs/corners/train310