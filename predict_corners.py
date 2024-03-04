import cv2
import numpy as np
from ultralytics import YOLO

#model = YOLO('/Users/patrick/Downloads/epoch30.pt')
model = YOLO('/Users/patrick/projects/ultralytics_git/runs/train4/weights/last.pt')

#model(source="input.mp4", show=True, conf=0.1, save=True, device='mps')
#results = model("dataset/train/images/0-1.png", conf=0.1)

colors = [(np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)) for _ in range(12)]

for i in range(1):
    for j in range(11):
        img = cv2.imread(f"/Users/patrick/projects/digicamp/dataset/train/images/{j}-{i}.png")
        results = model(img, verbose=False, conf=0.99)

        for r in results:
            
            boxes = r.boxes
            for b_i, box in enumerate(boxes):
                
                b = box.xyxy[0].numpy()  # get box coordinates in (left, top, right, bottom) format
                c = int(box.cls.numpy()[0])
                cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), colors[c], 2)
                
                cv2.putText(img, f'Class {c}', (10, 30+50*b_i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

            for corner in r.corners:
                for c_i in range(0, 12, 2):
                    cv2.circle(img, (int(corner[c_i][0]), int(corner[c_i][1])), 5, colors[c_i], -1)
                    cv2.circle(img, (int(corner[c_i+1][0]), int(corner[c_i+1][1])), 5, colors[c_i], -1)
                
        cv2.imshow(f'Class {j}, image {i}', img)
        cv2.waitKey(0)