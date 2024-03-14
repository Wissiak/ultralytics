import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

r1 = np.array([
    [0, 4],
    [1, 3],
    [1, 4],
    [2, 3],
    [1, 3],
    [2, 2],
    [1, 2],
    [2, 1],
    [2, 2],
    [3, 1],
    [1, 1],
    [2, 0]
])
r2 = np.array([
    [0, 4],
    [1, 3],
    [1, 4],
    [2, 3],
    [1, 3],
    [2, 2],
    [2, 3],
    [3, 2],
    [1, 2],
    [2, 1],
    [1, 1],
    [2, 0],
])
r3 = np.array([
    [0, 4],
    [1, 3],
    [1, 4],
    [2, 3],
    [1, 3],
    [2, 2],
    [1, 2],
    [2, 1],
    [1, 1],
    [2, 0],
    [2, 1],
    [3, 0],
])
r4 = np.array([
    [0, 4],
    [1, 3],
    [1, 4],
    [2, 3],
    [1, 3],
    [2, 2],
    [1, 2],
    [2, 1],
    [2, 2],
    [3, 1],
    [2, 1],
    [3, 0],
])
r5 = np.array([
    [0, 4],
    [1, 3],
    [1, 4],
    [2, 3],
    [2, 4],
    [3, 3],
    [1, 3],
    [2, 2],
    [1, 2],
    [2, 1],
    [1, 1],
    [2, 0],
])
r6 = np.array([
    [1, 4],
    [2, 3],
    [1, 3],
    [2, 2],
    [2, 3],
    [3, 2],
    [0, 2],
    [1, 1],
    [1, 2],
    [2, 1],
    [1, 1],
    [2, 0]
])
r7 = np.array([
    [0, 1],
    [1, 0],
    [1, 1],
    [2, 0],
    [1, 2],
    [2, 1],
    [2, 2],
    [3, 1],
    [2, 3],
    [3, 2],
    [3, 3],
    [4, 2]
])
r8 = np.array([
    [1, 4],
    [2, 3],
    [0, 3],
    [1, 2],
    [1, 3],
    [2, 2],
    [2, 3],
    [3, 2],
    [1, 2],
    [2, 1],
    [1, 1],
    [2, 0]
])
r9 = np.array([
    [2, 4],
    [3, 3],
    [0, 3],
    [1, 2],
    [1, 3],
    [2, 2],
    [2, 3],
    [3, 2],
    [1, 2],
    [2, 1],
    [1, 1],
    [2, 0]
])
r10 = np.array([
    [1, 4],
    [2, 3],
    [1, 3],
    [2, 2],
    [2, 3],
    [3, 2],
    [0, 2],
    [1, 1],
    [1, 2],
    [2, 1],
    [0, 1],
    [1, 0]
])
r11 = np.array([
    [1, 5],
    [2, 4],
    [1, 4],
    [2, 3],
    [0, 3],
    [1, 2],
    [1, 3],
    [2, 2],
    [0, 2],
    [1, 1],
    [0, 1],
    [1, 0],
])

all_pts = [r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11]

nc = 11
max_corners = 12
img_size = 640

model = YOLO('/Users/patrick/Downloads/trained-corners.pt')

colors = [(np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)) for _ in range(12)]

for i in range(nc):
    img = cv2.imread(f'/Users/patrick/projects/digicamp/dataset/test/images/{i}-2.png')
    ref_pts = all_pts[i]
   
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
                cv2.putText(img, str(i_c), (int(c[0]), int(c[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

            H, mask = cv2.findHomography(srcPoints=ref_pts, dstPoints=corner, method=cv2.RANSAC, ransacReprojThreshold=5.0)


            transformed_corners = cv2.perspectiveTransform(ref_pts.astype(np.float32).reshape(-1,1,2), H).squeeze()
            for c in transformed_corners:
                cv2.circle(img, (int(c[0]), int(c[1])), 5, (0, 0, 255), -1)
            cv2.imshow(f"würfelnetz-{i}-res-{r_i}.png", img)
            cv2.waitKey(0)


    #cv2.imwrite(f"test/würfelnetz-{i}.png", img)