from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    model = YOLO('./model/best.pt')
    metrics = model.val(data="./data/dataset/data.yaml")

    # Extract precision and recall for all classes
    precision = metrics.box.p
    recall = metrics.box.r 

    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
