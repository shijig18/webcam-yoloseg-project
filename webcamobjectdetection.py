
from ultralytics import YOLO
import cv2
import pyttsx3


# Initialize voice engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # speed of speech


model= YOLO("yolov8n.pt")
cap= cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    #YOLO Prediction
    results= model(frame)
    
    
        # Get detected class names
    class_names = results[0].names
    detected_classes = results[0].boxes.cls.tolist()

    # Check if person detected
    person_detected = False
    for cls_id in detected_classes:
        if class_names[int(cls_id)] == "person":
            person_detected = True
            break

    # Voice alert
    if person_detected:
        engine.say("Person detected")
        engine.runAndWait()
    
    
    
    # Draw results on frame
    annotated_frame= results[0].plot()
    
    
    
    #show output
    cv2.imshow('YOLO Detection', annotated_frame)
    
     # Exit key (press q)
    if cv2.waitKey(1) & 0xFF == ord('q'):
         break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
