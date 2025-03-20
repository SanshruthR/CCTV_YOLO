from ultralytics import YOLO
import cv2
import numpy as np
import gradio as gr
import torch

# Load YOLOv5n6 model
model = YOLO('yolov5n6.pt')

# Set the confidence threshold and IOU
model.conf = 0.25  # confidence threshold
model.iou = 0.45  # IOU threshold
model.agnostic = False
model.multi_label = False
model.max_det = 100  # max number of detections

# Low-resolution for inference 
#use frame_count += 3
#LOW_RES = (500, 600) 

#Faster inference but less accurate 
#LOW_RES = (320, 180) 

#Fastest inference but low accuracy, change frame_count to += 2
LOW_RES = (300, 300)

def detect_and_draw(frame):
    # Create low-res copy
    low_res_frame = cv2.resize(frame, LOW_RES)
    
    # Perform detection
    results = model(low_res_frame, verbose=False)

    # Scale bounding boxes
    scale_x = frame.shape[1] / LOW_RES[0]
    scale_y = frame.shape[0] / LOW_RES[1]

    # Draw bounding boxes on high-res frame
    for detection in results[0].boxes.data:
        x1, y1, x2, y2, conf, cls = detection
        x1, y1, x2, y2 = int(x1*scale_x), int(y1*scale_y), int(x2*scale_x), int(y2*scale_y)
        label = f"{results[0].names[int(cls)]} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return frame

# Define your stream URL
stream_url = "https://edge01.london.nginx.hdontap.com/hosb5/ng_showcase-coke_bottle-street_fixed.stream/chunklist_w464099566.m3u8"

def process_stream():
    cap = cv2.VideoCapture(stream_url)
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 2  #change to 3 for higher res
        if frame_count % 30 == 0:
            result = detect_and_draw(frame)
            result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            yield result_rgb

    cap.release()

iface = gr.Interface(
    fn=process_stream,
    inputs=None,
    outputs="image",
    live=True,
    title="Fast Real-time Object Detection with High-Res Output",
    description="Live stream processed with YOLOv5n6 on low-res frames, results drawn on high-res frames."
)

if __name__ == "__main__":
    if torch.cuda.is_available():
        model.to('cuda')
    #iface.queue()
    iface.launch()
