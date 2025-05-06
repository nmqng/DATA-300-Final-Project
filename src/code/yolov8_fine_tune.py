from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("./model/yolov8s.pt")

    # fine-tune the model
    model.train(
        data="./data.yaml",
        epochs=100,  # increased epochs from 50 to 100
        imgsz=768,  # increased image size from 640 to 768
        lr0=0.001,
        batch=16,  # reduced batch size from 32 to 16, then back to 32
        lrf=0.1,
        device='cuda'
    )