import cv2
from ultralytics import YOLO

# ===============================
# 1. LOAD SEGMENTATION MODEL
# ===============================
model = YOLO("yolov8n-seg.pt")  # segmentation model

# ===============================
# 2. OPEN WEBCAM
# ===============================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot access webcam")
    exit()

# ===============================
# 3. LOOP
# ===============================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Optional: flip for mirror view
    frame = cv2.flip(frame, 1)

    # ===============================
    # 4. YOLO SEGMENTATION
    # ===============================
    results = model(frame)

    # ===============================
    # 5. DRAW RESULTS
    # ===============================
    annotated_frame = results[0].plot()

    # ===============================
    # 6. DISPLAY
    # ===============================
    cv2.imshow("YOLO Segmentation", annotated_frame)

    # Exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()