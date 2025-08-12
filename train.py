from ultralytics import YOLO

# Load a YOLOv8 model (can be yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)
model = YOLO("yolov8s.pt")

# Train the model
# model.train(
#     data="datasets/deepfake-detection-2/data.yaml",  # Path to dataset YAML
#     epochs=50,
#     imgsz=640,
#     batch=16,
#     name="yolov8_deepfake",
# )
model.train(
    data="datasets/deepfake-detection-2/data.yaml",  # Path to dataset YAML
    epochs=5,
    imgsz=640,
    batch=16,
    name="yolov8_deepfake",
)