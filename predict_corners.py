import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO('/Users/patrick/Downloads/epoch19.pt')
#model = YOLO('/opt/homebrew/runs/detect/train54/weights/last.pt')

#model(source="input.mp4", show=True, conf=0.1, save=True, device='mps')
#results = model("dataset/train/images/0-1.png", conf=0.1)

colors = [(np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)) for _ in range(12)]

for i in range(1):
    for j in range(11):
        img = cv2.imread(f"/Users/patrick/projects/digicamp/dataset/test/images/{j}-{i}.png")
        results = model(img, verbose=False, conf=0.01)

        for r in results:
            
            boxes = r.boxes
            for box in boxes:
                
                b = box.xyxy[0].numpy()  # get box coordinates in (left, top, right, bottom) format
                c = int(box.cls.numpy()[0])
                img = cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), colors[c], 2)
            
            for corner in r.corners:
                for c in corner:
                    img = cv2.circle(img, (int(c[0]), int(c[1])), 5, (0, 0, 255), -1)
                
        cv2.imshow(f'Class {j}, image {i}', img)
        cv2.waitKey(0)