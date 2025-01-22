# config.py

# YOLO Model Configuration
MODEL_PATH = 'yolov8n.pt'  # Specify the YOLO model version (e.g., yolov5n, yolov8n, etc.)
CONF_THRESHOLD = 0.25      # Confidence threshold for object detection
IOU_THRESHOLD = 0.45       # Intersection-over-Union (IOU) threshold for non-max suppression

# ONNX Model Configuration (Optional)
ONNX_MODEL_PATH = 'model.onnx'  # Path to ONNX model file (leave empty if not using ONNX)

# Resolution Settings
LOW_RES_WIDTH = 320        # Low-resolution width for faster inference
LOW_RES_HEIGHT = 180       # Low-resolution height
CAM_RESOLUTION = (1280, 720)  # Camera resolution (width, height)

# Performance Settings
MAX_DETECTIONS = 100       # Maximum number of detections per frame
USE_GPU = True             # Set to False to force CPU inference
