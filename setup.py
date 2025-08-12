# Import and initialize Roboflow
from roboflow import Roboflow

# Replace with your own API key
rf = Roboflow(api_key="IMNRT0530J1AKw2XOEQn")

# Access the project (make sure workspace and project names are correct)
project = rf.workspace("ttetst").project("deepfake-detection-qdtvy")

# Select the version
version = project.version(2)

# Download in YOLOv8 format
dataset = version.download("yolov8")
