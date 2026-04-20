import torch
import torchvision.transforms as T
import cv2
import numpy as np
from torchvision import models
import ssl

# Fix SSL certificate verification issue on macOS
ssl._create_default_https_context = ssl._create_unverified_context

# ===============================
# 1. LOAD MODEL
# ===============================
def load_model():
    model = models.segmentation.deeplabv3_resnet50(pretrained=True)
    model.eval()
    return model

# ===============================
# 2. DEVICE SETUP
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model().to(device)

print(f"Using device: {device}")

# ===============================
# 3. IMAGE TRANSFORM
# ===============================
transform = T.Compose([
    T.ToPILImage(),
    T.Resize(320),  # Reduce for better FPS
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

# ===============================
# 4. COLOR MAP
# ===============================
def decode_segmap(label_mask):
    label_colors = np.array([
        (0, 0, 0),        # 0=background
        (128, 0, 0),      # 1
        (0, 128, 0),      # 2
        (128, 128, 0),    # 3
        (0, 0, 128),      # 4
        (128, 0, 128),    # 5
        (0, 128, 128),    # 6
        (128, 128, 128),  # 7
        (64, 0, 0),       # 8
        (192, 0, 0),      # 9
        (64, 128, 0),     # 10
        (192, 128, 0),    # 11
        (64, 0, 128),     # 12
        (192, 0, 128),    # 13
        (64, 128, 128),   # 14
        (192, 128, 128),  # 15
        (0, 64, 0),       # 16
        (128, 64, 0),     # 17
        (0, 192, 0),      # 18
        (128, 192, 0),    # 19
        (0, 64, 128)      # 20
    ])

    rgb = label_colors[label_mask]
    return rgb.astype(np.uint8)

# ===============================
# 5. SEGMENT FRAME FUNCTION
# ===============================
def segment_frame(frame):
    # Preprocess
    input_tensor = transform(frame).unsqueeze(0).to(device)

    # Prediction
    with torch.no_grad():
        output = model(input_tensor)['out'][0]

    # Get class for each pixel
    prediction = output.argmax(0).byte().cpu().numpy()

    # Convert to color mask
    seg_image = decode_segmap(prediction)

    # Resize back to original size
    seg_image = cv2.resize(seg_image, (frame.shape[1], frame.shape[0]))

    return seg_image

# ===============================
# 6. WEBCAM LOOP
# ===============================
def run_webcam():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Cannot access webcam")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip for mirror view (optional)
        frame = cv2.flip(frame, 1)

        # Get segmentation mask
        seg_image = segment_frame(frame)

        # Overlay mask
        overlay = cv2.addWeighted(frame, 0.6, seg_image, 0.4, 0)

        # Show outputs
        cv2.imshow("Original", frame)
        cv2.imshow("Segmentation Mask", seg_image)
        cv2.imshow("Overlay", overlay)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ===============================
# 7. RUN
# ===============================
if __name__ == "__main__":
    run_webcam()
