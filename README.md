This project demonstrates real-time image segmentation using a webcam feed. It uses a pretrained YOLOv8 segmentation model to detect and segment objects in live video.
The system captures frames continuously from the webcam and processes each frame through the model. The model identifies objects and generates segmentation masks, which are then overlaid on the original video stream. 
The output is displayed in real time, showing detected objects with colored masks and bounding boxes.
The model used in this project is a lightweight version (yolov8n-seg.pt) trained on the COCO dataset. It is optimized for speed and can run efficiently on systems without a GPU.
The main objective of this project is to understand how deep learning models can be applied to real-time computer vision tasks such as object detection and segmentation.
The webcam is initialized using OpenCV
Frames are captured continuously
Each frame is passed to the YOLOv8 segmentation model
The model predicts object masks and bounding boxes
Results are drawn on the frame
The processed frame is displayed in a window
