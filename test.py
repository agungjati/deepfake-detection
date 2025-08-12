from ultralytics import YOLO
import os

# Load model
model = YOLO("runs/detect/yolov8_deepfake/weights/best.pt")

# Run inference on images
results = model("datasets/deepfake-detection-2/test/images", 
save=True, 
save_txt=True,
conf=0.4,
iou=0.45
)

# Optional: print results summary
for result in results:
    print(f"\n🔍 Image: {result.path}")
    boxes = result.boxes
    names = result.names

    if boxes is None or len(boxes) == 0:
        print("❌ No objects detected.")
        continue

    for i, box in enumerate(boxes):
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        xyxy = box.xyxy[0].tolist()
        label = names[class_id]

        print(f"✅ Detected: {label}")
        print(f"    Confidence: {confidence:.2f}")
        print(f"    Box: {xyxy}")
