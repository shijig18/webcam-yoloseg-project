 Real-Time Object Segmentation using YOLOv8

This project demonstrates real-time object segmentation using the YOLOv8 segmentation model with a webcam feed. It uses OpenCV for video capture and Ultralytics YOLO for inference.

## Features

- Real-time webcam-based object segmentation
- Uses lightweight YOLOv8 Nano Segmentation model (`yolov8n-seg.pt`)
- Displays segmented objects with masks and bounding boxes
- Simple and easy-to-understand implementation

## Tech Stack

- Python
- OpenCV
- Ultralytics YOLOv8

## Project Structure

```
├── yolo_segment_webcam.py   # Main script for segmentation
├── README.md                # Project documentation
```

## How to Run

1. Clone or download this repository and open a terminal in the project folder.
2. Install the required libraries:
   ```
   pip install ultralytics opencv-python
   ```
3. Run the script:
   ```
   python yolo_segment_webcam.py
   ```
4. Allow camera access if your computer asks for permission. The model file `yolov8n-seg.pt` downloads automatically the first time you run the script.
5. A window opens showing your webcam feed with segmentation masks and bounding boxes. Press `q` to quit.

## Code Explanation

1. **Load Model**
   ```python
   model = YOLO("yolov8n-seg.pt")
   ```
   Loads the pre-trained YOLOv8 segmentation model.

2. **Capture Webcam**
   ```python
   cap = cv2.VideoCapture(0)
   ```
   Initializes webcam input.

3. **Perform Segmentation**
   ```python
   results = model(frame)
   ```
   Runs inference on each frame.

4. **Visualize Output**
   ```python
   annotated_frame = results[0].plot()
   ```
   Draws segmentation masks and bounding boxes.

5. **Display Frame**
   ```python
   cv2.imshow("YOLO Segmentation", annotated_frame)
   ```
   Shows the annotated frame in a window.
