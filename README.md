### Abstract
This project is about real-time vehicles detection using the YOLOv8s model for traffic monitoring. The system detects 6 types of vehicles: bus, truck, motorbike, bicycle, and person, in video frames with minimal delay. The model is trained on a dataset of over 5,800 augmented images, resized to 768×768 pixels to ensure detection at further distances. Streamlink and OpenCV libraries are used to retrieve live videos from YouTube and handle video input and frame processing. A vehicle counter is used determine the street status (CROWDED or NOT) based on a predefined threshold (10 vehicles at a given frame). The model achieved a precision of 0.935, recall of 0.94, and mAP50 of 0.96, indicating strong performance. However, the limitation is lack of low-light and different weather conditions of the dataset.

### Some notes
- File ```./src/code/real_time_predict.py``` is used for predicting live videos YouTube (real-time detection).
- File ```./src/code/real_time_predict_multi.py``` is used for predicting many (3) live videos from YouTube at once, then it will show when/which video is currently showing CROWDED status.
