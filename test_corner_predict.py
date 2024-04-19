import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

nc = 11
max_corners = 12
img_size = 640

if __name__ == "__main__":
        
    model = YOLO('/Users/patrick/Downloads/epoch140.pt')

    colors = [(np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)) for _ in range(12)]

    for j in range(10):
        for i in range(nc):
            img = cv2.imread(f'/Users/patrick/projects/digicamp/dataset/train/images/{i}-{j}.png')
        
            results = model.predict([img], conf=0.5)

            for r_i, r in enumerate(results):
                annotator = Annotator(img)
                
                boxes = r.boxes
                for box in boxes:
                    b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
                    c = box.cls
                    annotator.box_label(b, model.names[int(c)], colors[c.cpu().numpy().astype(np.uint8)[0]])
                img = annotator.result()

                corners = r.corners
                for corner in corners.cpu().numpy().astype(np.int32):
                    for i_c, c in enumerate(corner):
                        cv2.circle(img, (int(c[0]), int(c[1])), 5, (0, 255, 0), -1)
                        y_offset = -10 if i_c % 2 == 0 else 10
                        y = int(c[1]) -10
                        cv2.putText(img, str(i_c), (int(c[0]), y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                    cv2.imshow(f"cube-net-{i}-{j}-res-{r_i}.png", img)
                    cv2.waitKey(0)


            #cv2.imwrite(f"test/würfelnetz-{i}.png", img)