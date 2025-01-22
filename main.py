from ultralytics import YOLO
import cv2
import numpy as np
import gradio as gr
import torch
from bettercam import Camera
import config  # Import configurations from config.py
import onnx
from onnxruntime import InferenceSession, SessionOptions

# Load YOLO model dynamically
try:
    model = YOLO(config.MODEL_PATH)
    print(f"Loaded model: {config.MODEL_PATH}")
except Exception as e:
    raise ValueError(f"Error loading model from {config.MODEL_PATH}: {e}")

# Configure model parameters
model.conf = config.CONF_THRESHOLD
model.iou = config.IOU_THRESHOLD
model.agnostic = False
model.multi_label = False
model.max_det = 100  # Max number of detections

# Define low resolution for faster inference
LOW_RES = (config.LOW_RES_WIDTH, config.LOW_RES_HEIGHT)

# Load ONNX model if specified in the config
onnx_model_path = getattr(config, "ONNX_MODEL_PATH", None)
onnx_session = None
if onnx_model_path:
    print(f"Loading ONNX model from {onnx_model_path}...")
    options = SessionOptions()
    try:
        onnx_session = InferenceSession(onnx_model_path, options)
        print("ONNX model loaded successfully!")
    except Exception as e:
        raise ValueError(f"Failed to load ONNX model: {e}")

# Function to detect objects and draw bounding boxes
def detect_and_draw(frame):
    # Resize frame to low resolution for inference
    low_res_frame = cv2.resize(frame, LOW_RES)

    # Perform detection using the YOLO model
    results = model(low_res_frame, verbose=False)

    # Scale bounding boxes back to high resolution
    scale_x = frame.shape[1] / LOW_RES[0]
    scale_y = frame.shape[0] / LOW_RES[1]

    # Draw bounding boxes on the original high-res frame
    for detection in results[0].boxes.data:
        x1, y1, x2, y2, conf, cls = detection
        x1, y1, x2, y2 = int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)
        label = f"{results[0].names[int(cls)]} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return frame

# Function to process the video stream using BetterCam
def process_stream():
    # Initialize BetterCam
    cam = Camera()
    cam.set_resolution(*config.CAM_RESOLUTION)  # Set desired resolution
    cam.start()

    try:
        while True:
            # Capture frame from BetterCam
            frame = cam.get_latest_frame()
            if frame is None:
                continue

            # Perform object detection and drawing
            result = detect_and_draw(frame)

            # Convert BGR to RGB for display
            result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            yield result_rgb
    finally:
        cam.stop()

# Gradio interface setup
iface = gr.Interface(
    fn=process_stream,
    inputs=None,
    outputs="image",
    live=True,
    title="Dynamic YOLO and ONNX Real-time Object Detection",
    description="Live stream processed with dynamic YOLO or ONNX model using BetterCam for efficient video capture."
)

if __name__ == "__main__":
    # Check for GPU availability and use CUDA if available
    if torch.cuda.is_available():
        model.to('cuda')
    iface.launch()
