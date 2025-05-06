import streamlink
import cv2
import time
from ultralytics import YOLO

# youtube_url = 'https://www.youtube.com/watch?v=DLmn7f9SJ5A'
# youtube_url = 'https://www.youtube.com/watch?v=_0wPODlF9wU'
# youtube_url = 'https://www.youtube.com/watch?v=ByED80IKdIU'
# youtube_url = 'https://www.youtube.com/watch?v=6dp-bvQ7RWo' # many people
# youtube_url = 'https://www.youtube.com/watch?v=9SLt3AT0rXk' # light traffic
youtube_url = 'https://www.youtube.com/watch?v=5WN2PJ_Qxjs'   # good prediction
# youtube_url = 'https://www.youtube.com/watch?v=KSsfLxP-A9g' # pretty crowded source

conf_threshold = 0.5
crowd_threshold = 10  # Number of vehicles considered "crowded"
vehicle_classes = ['bicycle', 'bus', 'car', 'motorcycle', 'person', 'truck']

# Load YOLOv8 model
# model = YOLO('./model/best.pt')
model = YOLO('./model/yolov8s.pt')
class_names = model.names

# Get YouTube stream URL
streams = streamlink.streams(youtube_url)
if streams:
    stream_url = streams.get('720p', streams['best']).url

# Open the video stream
cap = cv2.VideoCapture(stream_url)

# FPS tracking
frame_count = 0
start_time = time.time()
fps_text = ""

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection (0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck')
    results = model(frame, conf=conf_threshold, classes=[0, 1, 2, 3, 5, 7])[0]

    # Count vehicle objects
    vehicle_count = 0
    for box in results.boxes:
        cls_id = int(box.cls[0])
        cls_name = class_names[cls_id]
        if cls_name in vehicle_classes:
            vehicle_count += 1

    # Determine crowd status
    crowd_status = "CROWDED" if vehicle_count > crowd_threshold else "Not Crowded"
    status_color = (0, 0, 255) if crowd_status == "CROWDED" else (0, 255, 0)

    # Annotate frame
    annotated_frame = results.plot()
    cv2.putText(annotated_frame, f'Vehicles: {vehicle_count}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.putText(annotated_frame, f'Status: {crowd_status}', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, status_color, 3)

    frame_count += 1
    elapsed_time = time.time() - start_time
    if elapsed_time >= 2:
        fps_text = f"FPS: {frame_count / elapsed_time:.2f}"
        frame_count = 0
        start_time = time.time()

    cv2.putText(annotated_frame, fps_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 100), 2)
    cv2.imshow("YouTube Traffic Monitor", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
