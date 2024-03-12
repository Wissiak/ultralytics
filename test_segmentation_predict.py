from ultralytics import YOLO
import random
from ultralytics import YOLO
from ultralytics.utils.ops import scale_image
import cv2
import numpy as np
from sklearn.cluster import KMeans
pts1_w = np.array([
    [0, 4, 0.1], # 0
    [0, 3, 0.1], # 1
    [1, 3, 0.1], # 2
    [1, 0, 0.1], # 3
    [2, 0, 0.1], # 4
    [2, 1, 0.1], # 5
    [3, 1, 0.1], # 6
    [3, 2, 0.1], # 7
    [2, 2, 0.1], # 8
    [2, 4, 0.1], # 9
])

# def predict_on_image(model, img, conf):
#     result = model(img, conf=conf)[0]

#     # detection
#     # result.boxes.xyxy   # box with xyxy format, (N, 4)
#     cls = result.boxes.cls.cpu().numpy()    # cls, (N, 1)
#     probs = result.boxes.conf.cpu().numpy()  # confidence score, (N, 1)
#     boxes = result.boxes.xyxy.cpu().numpy()   # box with xyxy format, (N, 4)

#     # segmentation
#     masks = result.masks.data.cpu().numpy()     # masks, (N, H, W)
#     masks = np.moveaxis(masks, 0, -1) # masks, (H, W, N)
#     # rescale masks to original image
#     masks = scale_image(masks, result.masks.orig_shape)
#     masks = np.moveaxis(masks, -1, 0) # masks, (N, H, W)

#     return boxes, masks, cls, probs


def overlay(image, mask, color, alpha, resize=None):
    """Combines image and its segmentation mask into a single image.
    https://www.kaggle.com/code/purplejester/showing-samples-with-segmentation-mask-overlay

    Params:
        image: Training image. np.ndarray,
        mask: Segmentation mask. np.ndarray,
        color: Color for segmentation mask rendering.  tuple[int, int, int] = (255, 0, 0)
        alpha: Segmentation mask's transparency. float = 0.5,
        resize: If provided, both image and its mask are resized before blending them together.
        tuple[int, int] = (1024, 1024))

    Returns:
        image_combined: The combined image. np.ndarray

    """
    #color = color[::-1]
    colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
    colored_mask = np.moveaxis(colored_mask, 0, -1)
    masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
    image_overlay = masked.filled()

    if resize is not None:
        image = cv2.resize(image.transpose(1, 2, 0), resize)
        image_overlay = cv2.resize(image_overlay.transpose(1, 2, 0), resize)

    image_combined = cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)

    return image_combined



def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

# Load a model
model = YOLO('/home/patrick.wissiak/work/runs/train12/weights/best.pt')  # load a custom model

class_names = model.names
print('Class Names: ', class_names)
colors = [[random.randint(0, 255) for _ in range(3)] for _ in class_names]

n_corners = [10, 10, 8, 10, 8, 12, 12, 12, 10, 12, 8]

for j in range(11):
    for i in range(8):
        # load image by OpenCV like numpy.array
        img = cv2.imread(f'/home/patrick.wissiak/work/dataset/train/images/{j}-{i}.png')

        h, w, _ = img.shape
        results = model.predict(img, stream=False, verbose=False)
        # print(results)
        for r in results:
            boxes = r.boxes  # Boxes object for bbox outputs
            masks = r.masks  # Masks object for segment masks outputs
            probs = r.probs  # Class probabilities for classification outputs

        if masks is not None:
            masks = masks.data.cpu().numpy()
            for seg, box in zip(masks, boxes):

                seg = cv2.resize(seg, (w, h))
                img = overlay(img, seg, colors[int(box.cls)], 0.4)
                
                xmin = int(box.data[0][0])
                ymin = int(box.data[0][1])
                xmax = int(box.data[0][2])
                ymax = int(box.data[0][3])
                
                plot_one_box([xmin, ymin, xmax, ymax], img, colors[int(box.cls)], f'{class_names[int(box.cls)]} {float(box.conf):.3}')

                # draw corners
                min_distance = min((xmax-xmin)/4, np.sqrt(ymax-ymin)/4)
                desired_corners = n_corners[int(box.cls)]
                contours, _ = cv2.findContours(seg.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                largest_contour = max(contours, key=cv2.contourArea)
                epsilon = 0.01 * cv2.arcLength(largest_contour, True)
                approximated_contour = cv2.approxPolyDP(largest_contour, epsilon, True)

                corners = approximated_contour.squeeze()
                # filtered_corners = [corners[0]]
                # for corner in corners[1:]:
                #     if all(np.linalg.norm(np.array(corner) - np.array(existing_corner)) > min_distance for existing_corner in filtered_corners):
                #         filtered_corners.append(corner)

                # filtered_corners = np.array(filtered_corners)[:desired_corners]
                kmeans = KMeans(n_clusters=min(desired_corners, len(corners)), random_state=0).fit(corners)
                cluster_centers = kmeans.cluster_centers_

                if cluster_centers.shape[0] != desired_corners:
                    print(f"Got unprecise corners for {j}-{i}")

                image_with_corners = img.copy()
                for c_i, corner in enumerate(cluster_centers):
                    cv2.circle(image_with_corners, tuple(map(int, corner.ravel())), 5, (0, 0, 255), -1)
                    #cv2.putText(image_with_corners, str(c_i), tuple(map(int, corner.ravel())), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                H, mask = cv2.findHomography(cluster_centers, pts1_w[:, :2], cv2.RANSAC, 20.0)
                if not all(mask):
                    print(f"Got bad homography for {j}-{i}")
                
                transformed_corners = cv2.perspectiveTransform(pts1_w[:,:2].reshape(-1,1,2)*640, H)
                cv2.polylines(image_with_corners, [np.int32(transformed_corners)], True, (0, 0, 255), 3, cv2.LINE_AA)

                # Display or save the results as needed

                # Saving the image
                cv2.imwrite(f'segmentation_result_{j}_{i}.png', image_with_corners)