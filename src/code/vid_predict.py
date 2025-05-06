import os
from ultralytics import YOLO

input_dir = './data/input_vid'
output_dir = './data/output_vid'

# model = YOLO('./model/best.pt')
model = YOLO('./model/yolov8s.pt')

os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.lower().endswith('.mp4'):
        input_path = os.path.join(input_dir, filename)

        model.predict(
            source=input_path,
            save=True,
            # next line is the class id for vehicle classes when using yolov8 model
            # uncomment next 2 lines if using best model
            classes=[0, 1, 2, 3, 5, 7], 
            conf=0.5,  # confidence threshold when using yolov8 model
            save_txt=False,
            save_conf=True,
            project=output_dir,
            name='',
            exist_ok=True
        )
