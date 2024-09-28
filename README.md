# Fast Real-time Object Detection with High-Res Output

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![YOLOv5](https://img.shields.io/badge/YOLOv5n6-6.1-orange?style=for-the-badge&logo=YOLO&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-Enabled-76B900?style=for-the-badge&logo=nvidia&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-3.4.2-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-3.1.4-blueviolet?style=for-the-badge&logo=gradio&logoColor=white)
[![Deployed on Hugging Face](https://img.shields.io/badge/Deployed%20on-Hugging%20Face-yellow?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/spaces/Sanshruth/CCTV_YOLO?embed=true)
![image](https://github.com/user-attachments/assets/2cbae46f-e9de-48db-96f8-d439536ad703)


## Overview

Live stream of Las Vegas sidewalk traffic cam, processed with YOLOv5n6 on low-resolution frames, with results drawn on high-resolution frames.

This project demonstrates **real-time object detection** using the YOLOv5n6 model with **low-resolution inference** for high-speed processing, while drawing the results on **high-resolution frames**. The object detection pipeline is deployed as a Gradio app and streams live data from an external camera feed.

### Features

- **YOLOv5n6 Model**: Pre-trained object detection model optimized for speed and accuracy.
- **Low-resolution Processing**: Efficiently processes frames in low resolution (320x180) while mapping results to high-res images.
- **Gradio Interface**: Interactive Gradio interface with real-time video stream processing.
- **CUDA Support**: Optimized for CUDA-enabled GPUs, ensuring fast inference times.

### Model Details

- **Model**: YOLOv5n6 (`yolov5n6.pt`)
- **Confidence Threshold**: 0.25
- **IOU Threshold**: 0.45
- **Max Detections**: 100 objects per frame

### How It Works

The pipeline processes a live video stream, detecting objects in each frame using YOLOv5n6. Bounding boxes are drawn on the high-resolution frames based on detections from the low-resolution inference.

### Usage

1. Clone the repository and install the dependencies:
    ```bash
    git clone https://github.com/SanshruthR/CCTV_YOLO.git
    cd cctv-yolo
    pip install -r requirements.txt
    ```

2. Run the script:
    ```bash
    python app.py
    ```

3. Access the Gradio interface and view the live stream processed with YOLOv5n6.

### Deployment

This project is deployed on **Hugging Face Spaces**. You can interact with the app via the following link:

[Live Demo on Hugging Face](https://huggingface.co/spaces/Sanshruth/CCTV_YOLO?embed=true)

### License

This project is licensed under the MIT License.
