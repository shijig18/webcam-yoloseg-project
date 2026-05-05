Real-Time Object Segmentation using YOLOv8

This project demonstrates real-time object segmentation using the YOLOv8 segmentation model with a webcam feed. It uses OpenCV for video capture and Ultralytics YOLO for inference.

Features
      
    Real-time webcam-based object segmentation
    Uses lightweight YOLOv8 Nano Segmentation model (yolov8n-seg.pt)
    Displays segmented objects with masks and bounding boxes
    Simple and easy-to-understand implementation

Tech Stack

    Python
    OpenCV
    Ultralytics YOLOv8

Project Structure

    ├── yolo_segment_webcam.py          # Main script for segmentation
    ├── README.md                       # Project documentation

Code Explanation

1. Load Model
   
      model = YOLO("yolov8n-seg.pt")

   Loads the pre-trained YOLOv8 segmentation model.

3. Capture Webcam
   
      cap = cv2.VideoCapture(0)

   Initializes webcam input.

5. Perform Segmentation
   
      results = model(frame)

    Runs inference on each frame.

7. Visualize Output
   
      annotated_frame = results[0].plot()

   Draws segmentation masks and bounding boxes.

9. Display Frame
    
      cv2.imshow("YOLO Segmentation", annotated_frame)
