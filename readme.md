# YOLO Real-time Object Detection with BetterCam

This project provides a real-time object detection system using **YOLO (You Only Look Once)** models and **BetterCam** for efficient video stream capture. It supports any YOLO version, allowing flexibility for various detection needs. 

## Features
- **Dynamic YOLO Model Support**: Easily switch between YOLOv5, YOLOv8, or other YOLO models.
- **Efficient Video Capture**: Utilizes BetterCam for low-latency, high-quality frame capture.
- **Configurable Settings**: Adjust thresholds, resolutions, and model paths via a configuration file.
- **Real-time Performance**: Processes video streams in real-time with bounding box visualization.

## How It Works
1. **Video Input**: Captures video frames using BetterCam.
2. **Object Detection**: Performs object detection on resized low-resolution frames.
3. **Visualization**: Scales the detection results back to the original resolution and overlays bounding boxes on the video.

## Installation

### Prerequisites
- Python 3.8 or higher
- A compatible YOLO model (e.g., `yolov5n.pt`, `yolov8n.pt`)
- A GPU with CUDA support (optional, but recommended)

### Dependencies
Install the required Python packages:
```
pip install ultralytics opencv-python numpy gradio torch bettercam cupy-cuda11x onnx onnxruntime-gpu onnx-simplifier onnxruntime
```

## Configuration

- Create a `config.py` file in the project directory to define your settings:

```
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
```

## Usage

1. click the green button to download the zip file.

2. make sure to install the requirements needed to run the project.

3. run the program

```
python main.py
```
- Open the Gradio interface in your browser. The live video stream will display with object detection results.

## Customization

- To switch YOLO models, update the `MODEL_PATH` in `config.py`.

- Adjust detection thresholds (`CONF_THRESHOLD` and `IOU_THRESHOLD`) in `config.py` for your use case.

- Change resolutions (`LOW_RES_WIDTH`, `LOW_RES_HEIGHT`, `CAM_RESOLUTION`) to optimize performance.


## Supported YOLO Versions
- YOLOv5
- YOLOv7
- YOLOv8

## Exporting Yolo 
- [Repo: Exporting Yolo](https://github.com/KernFerm/exporting-YOLO)

## Contributions

Contributions are welcome! If you have ideas for improvement, feel free to submit a pull request or open an issue.


## License
This project is licensed under the `MIT License`.


## Acknowledgments

- [Ultralytics](https://ultralytics.com/) for the YOLO models.

- [BetterCam](https://github.com/RootKit-Org/BetterCam) for efficient video capture.

- [Gradio](https://gradio.app/) for the user-friendly interface.

- Forked from [SanshruthR/CCTV_YOLO](https://github.com/SanshruthR/CCTV_YOLO)


### 1. **Download the NVIDIA CUDA Toolkit 11.8**

First, download the CUDA Toolkit 11.8 from the official NVIDIA website:

👉 [Nvidia CUDA Toolkit 11.8 - DOWNLOAD HERE](https://developer.nvidia.com/cuda-11-8-0-download-archive)

### 2. **Install the CUDA Toolkit**

- After downloading, open the installer (`.exe`) and follow the instructions provided by the installer.
- Make sure to select the following components during installation:
  - CUDA Toolkit
  - CUDA Samples
  - CUDA Documentation (optional)

### 3. **Verify the Installation**

- After the installation completes, open the `cmd.exe` terminal and run the following command to ensure that CUDA has been installed correctly:
```
nvcc --version
```
This will display the installed CUDA version.

### **4. Install Cupy**
Run the following command in your terminal to install Cupy:
```
pip install cupy-cuda11x
```

## 5. CUDNN Installation 🧩
Download cuDNN (CUDA Deep Neural Network library) from the NVIDIA website:

👉 [Download CUDNN](https://developer.nvidia.com/downloads/compute/cudnn/secure/8.9.6/local_installers/11.x/cudnn-windows-x86_64-8.9.6.50_cuda11-archive.zip/). (Requires an NVIDIA account – it's free).

## 6. Unzip and Relocate 📁➡️
Open the `.zip` cuDNN file and move all the folders/files to the location where the CUDA Toolkit is installed on your machine, typically:

```
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8
```


## 7. Get TensorRT 8.6 GA 🔽
Download [TensorRT 8.6 GA](https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/secure/8.6.1/zip/TensorRT-8.6.1.6.Windows10.x86_64.cuda-11.8.zip).

## 8. Unzip and Relocate 📁➡️
Open the `.zip` TensorRT file and move all the folders/files to the CUDA Toolkit folder, typically located at:

```
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8
```


## 9. Python TensorRT Installation 🎡
Once all the files are copied, run the following command to install TensorRT for Python:

```
pip install "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\python\tensorrt-8.6.1-cp311-none-win_amd64.whl"
```

🚨 **Note:** If this step doesn’t work, double-check that the `.whl` file matches your Python version (e.g., `cp311` is for Python 3.11). Just locate the correct `.whl` file in the `python` folder and replace the path accordingly.

## 10. Set Your Environment Variables 🌎
Add the following paths to your environment variables:
- `system path`
```
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\lib
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\libnvvp
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin
```

# Setting Up CUDA 11.8 with cuDNN on Windows

Once you have CUDA 11.8 installed and cuDNN properly configured, you need to set up your environment via `cmd.exe` to ensure that the system uses the correct version of CUDA (especially if multiple CUDA versions are installed).

## Steps to Set Up CUDA 11.8 Using `cmd.exe`

### 1. Set the CUDA Path in `cmd.exe`

You need to add the CUDA 11.8 binaries to the environment variables in the current `cmd.exe` session.

Open `cmd.exe` and run the following commands:
- DO each one `Separately`
```
set PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin;%PATH%
set PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\libnvvp;%PATH%
set PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\extras\CUPTI\lib64;%PATH%
```
These commands add the CUDA 11.8 binary, lib, and CUPTI paths to your system's current session. Adjust the paths as necessary depending on your installation directory.

2. Verify the CUDA Version
After setting the paths, you can verify that your system is using CUDA 11.8 by running:
```
nvcc --version
```
This should display the details of CUDA 11.8. If it shows a different version, check the paths and ensure the proper version is set.

3. **Set the Environment Variables for a Persistent Session**
If you want to ensure CUDA 11.8 is used every time you open `cmd.exe`, you can add these paths to your system environment variables permanently:

1. Open `Control Panel` -> `System` -> `Advanced System Settings`.
Click on `Environment Variables`.
Under `System variables`, select `Path` and click `Edit`.
Add the following entries at the top of the list:
```
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\libnvvp
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\extras\CUPTI\lib64
```
This ensures that CUDA 11.8 is prioritized when running CUDA applications, even on systems with multiple CUDA versions.

4. **Set CUDA Environment Variables for cuDNN**
If you're using cuDNN, ensure the `cudnn64_8.dll` is also in your system path:
```
set PATH=C:\tools\cuda\bin;%PATH%
```
This should properly set up CUDA 11.8 to be used for your projects via `cmd.exe`.

