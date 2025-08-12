from ultralytics import YOLO

# Load your trained model
model = YOLO("runs/detect/yolov8_deepfake5/weights/best.pt")

# Run validation
metrics = model.val()
print(metrics)
