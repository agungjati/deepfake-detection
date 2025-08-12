from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from ultralytics import YOLO
import shutil
import os
from PIL import Image
import io

# Load the trained YOLOv8 model
model = YOLO("runs/detect/yolov8_deepfake/weights/best.pt")  # Replace with your model path

app = FastAPI()

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    # Save uploaded image to a temporary file
    temp_file_path = f"temp_{file.filename}"
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Run detection
    results = model(temp_file_path, conf=0.4 )

    # Save annotated image to memory (instead of disk)
    annotated_img_path = results[0].save(filename="annotated.jpg")

    # Open the annotated image
    image = Image.open("annotated.jpg")
    image_io = io.BytesIO()
    image.save(image_io, format="JPEG")
    image_io.seek(0)

    # Clean up temporary files
    os.remove(temp_file_path)
    os.remove("annotated.jpg")

    # Return image with annotations
    return StreamingResponse(image_io, media_type="image/jpeg")
