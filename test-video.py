import cv2
from ultralytics import YOLO
from test_corner_predict import all_pts
import numpy as np

# Load the YOLOv8 model
model = YOLO('/Users/patrick/Downloads/trained-corners.pt')

# Open the video file
video_path = "test-video.mov"
cap = cv2.VideoCapture(0)

resize=False

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if not success:
        break

    if resize:
        orig_size = (frame.shape[1], frame.shape[0])
        frame = cv2.resize(frame, (640, 640)) 
    #else:
    #    frame = frame[:640, :640]

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        if len(results[0].boxes) > 0:
            
            cls = results[0].boxes[0].cls.cpu().numpy().astype(np.uint8)[0]

            ref_pts = all_pts[cls]

            corners = results[0].corners
            for corner in corners.cpu().numpy().astype(np.int32):
                for i_c, c in enumerate(corner):
                    cv2.circle(annotated_frame, (int(c[0]), int(c[1])), 5, (0, 255, 0), -1)
                    y_offset = -10 if i_c % 2 == 0 else 10
                    cv2.putText(annotated_frame, str(i_c), (int(c[0]), int(c[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

                H, mask = cv2.findHomography(srcPoints=ref_pts, dstPoints=corner, method=cv2.RANSAC, ransacReprojThreshold=5.0)


                transformed_corners = cv2.perspectiveTransform(ref_pts.astype(np.float32).reshape(-1,1,2), H).squeeze()
                for c in transformed_corners:
                    cv2.circle(annotated_frame, (int(c[0]), int(c[1])), 5, (0, 0, 255), -1)

        # Display the annotated frame
        if resize:
            annotated_frame = cv2.resize(annotated_frame, orig_size) 
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()