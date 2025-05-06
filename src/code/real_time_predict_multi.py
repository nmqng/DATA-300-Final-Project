import streamlink
import cv2
import time
import threading
import numpy as np
from ultralytics import YOLO
from queue import Queue

youtube_urls = [
    'https://www.youtube.com/watch?v=5WN2PJ_Qxjs',
    'https://www.youtube.com/watch?v=6dp-bvQ7RWo',
    'https://www.youtube.com/watch?v=9SLt3AT0rXk',
]

conf_threshold = 0.5
crowd_threshold = 10 
vehicle_classes = ['bicycle', 'bus', 'car', 'motorcycle', 'person', 'truck']

# Load model
model = YOLO('./model/yolov8s.pt')
class_names = model.names 

# Shared data structure for frames and status
frame_queues = {url: Queue(maxsize=1) for url in youtube_urls}
status_dict = {url: {"status": "Not Crowded", "vehicle_count": 0,
                     "active": True, "fps": 0.0} for url in youtube_urls}

status_lock = threading.Lock()


def process_stream(url, stream_idx):
    """Detecting single YouTube stream"""
    try:
        # Get YouTube stream URL
        streams = streamlink.streams(url)
        if not streams:
            print(f"No streams available for {url}")
            with status_lock:
                status_dict[url]["active"] = False
            return
        # Using 480p to reduce load
        stream_url = streams.get('480p', streams.get('720p', streams['best'])).url  

        # Open the video stream
        cap = cv2.VideoCapture(stream_url)
        if not cap.isOpened():
            print(f"{url} failed")
            with status_lock:
                status_dict[url]["active"] = False
            return

        # FPS tracking
        frame_count = 0
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                with status_lock:
                    status_dict[url]["active"] = False
                break

            try:
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

                # Update status dictionary
                with status_lock:
                    status_dict[url]["status"] = crowd_status
                    status_dict[url]["vehicle_count"] = vehicle_count
                    status_dict[url]["active"] = True

                # Annotate frame
                annotated_frame = results.plot()
                cv2.putText(annotated_frame, f'Vehicles: {vehicle_count}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                cv2.putText(annotated_frame, f'Status: {crowd_status}', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, status_color, 3)

                # FPS calculation
                frame_count += 1
                elapsed_time = time.time() - start_time
                if elapsed_time >= 2:
                    fps = frame_count / elapsed_time
                    with status_lock:
                        status_dict[url]["fps"] = fps
                    frame_count = 0
                    start_time = time.time()
                    fps_text = f"FPS: {fps:.2f}"
                else:
                    fps_text = f"FPS: {status_dict[url]['fps']:.2f}"

                cv2.putText(annotated_frame, fps_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 100), 2)

                # Store the latest frame in the queue
                if frame_queues[url].full():
                    frame_queues[url].get()  # Remove old frame
                frame_queues[url].put(annotated_frame)

            except Exception as e:
                print(e)
                continue

            # Check for quit signal
            with status_lock:
                if not status_dict[url]["active"]:
                    break

        cap.release()

    except Exception as e:
        print(e)
        with status_lock:
            status_dict[url]["active"] = False


def main():
    # Start a thread for each stream
    stream_threads = []
    for idx, url in enumerate(youtube_urls):
        thread = threading.Thread(target=process_stream, args=(url, idx))
        thread.daemon = True  # Exit program even if threads are running
        thread.start()
        stream_threads.append(thread)

    try:
        while any(thread.is_alive() for thread in stream_threads):
            for idx, url in enumerate(youtube_urls):
                try:
                    if not frame_queues[url].empty():
                        frame = frame_queues[url].get_nowait()
                        cv2.imshow(f"Stream {idx + 1}: {url[-12:]}", frame)
                except Exception as e:
                    print(e)
                    continue

            # Create and display status window
            status_img = np.zeros((100 + 50 * len(youtube_urls), 600, 3), dtype=np.uint8)
            with status_lock:
                is_any_crowded = any(
                    info["status"] == "CROWDED" and info["active"] for info in status_dict.values())
                if not is_any_crowded:
                    cv2.putText(status_img, "All Streams Clear", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                else:
                    cv2.putText(status_img, "Crowded Streams Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

                for i, (url, info) in enumerate(status_dict.items()):
                    if info["active"]:
                        color = (0, 0, 255) if info["status"] == "CROWDED" else (0, 255, 0)
                        text = f"Stream {i + 1} ({url[-12:]}): {info['status']} ({info['vehicle_count']} vehicles)"
                    else:
                        color = (128, 128, 128)  # Gray for inactive streams
                        text = f"Stream {i + 1} ({url[-12:]}): Inactive"
                    cv2.putText(status_img, text, (50, 100 + i * 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            cv2.imshow("Crowd Status Summary", status_img)

            # Handle keyboard input
            if cv2.waitKey(1) & 0xFF == ord('q'):
                with status_lock:
                    for url in youtube_urls:
                        status_dict[url]["active"] = False
                break

    except KeyboardInterrupt:
        print("Shut down")
        
    finally:
        # Clean up
        with status_lock:
            for url in youtube_urls:
                status_dict[url]["active"] = False
        for url in youtube_urls:
            cv2.destroyWindow(f"Stream {youtube_urls.index(url) + 1}: {url[-12:]}")
        cv2.destroyWindow("Crowd Status Summary")
        cv2.destroyAllWindows()
        for thread in stream_threads:
            thread.join()


if __name__ == "__main__":
    main()
