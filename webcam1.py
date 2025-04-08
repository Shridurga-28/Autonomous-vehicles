import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch
from bisenetv2 import BiSeNet  # Make sure bisenetv2.py is in the same directory

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load model
model = BiSeNet(num_classes=256).to(device)
model.load_state_dict(torch.load("bisenet_model.pth", map_location=device), strict=False)
model.eval()

# Define image transform
transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# Predict segmentation mask
def predict(frame):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = transform(image=image)["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image).argmax(dim=1).cpu().numpy()[0]

    return output

# Determine lane direction
def get_lane_direction(mask):
    height, width = mask.shape
    lane_points = np.column_stack(np.where(mask > 0))  # (y, x)

    if len(lane_points) < 500:
        return "No lane detected"

    x = lane_points[:, 1]
    y = lane_points[:, 0]

    poly = np.polyfit(y, x, 2)
    y_eval = np.max(y)
    A = poly[0]
    B = poly[1]

    radius = ((1 + (2*A*y_eval + B)**2)**1.5) / np.abs(2*A)

    if radius > 3000:
        return "Straight"
    elif A > 0:
        return "Right Turn"
    else:
        return "Left Turn"

# Real-time video segmentation from webcam
def segment_video():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Cannot open webcam")
        return

    print("Starting webcam feed. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        segmentation_map = predict(frame)
        mask_resized = cv2.resize(segmentation_map, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

        direction = get_lane_direction(mask_resized)
        mask_colored = cv2.applyColorMap((mask_resized * 10).astype(np.uint8), cv2.COLORMAP_JET)
        overlayed_frame = cv2.addWeighted(frame, 0.6, mask_colored, 0.4, 0)

        cv2.putText(overlayed_frame, f"Direction: {direction}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("Lane Detection (Press 'q' to quit)", overlayed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Webcam session ended.")

# Run the function
segment_video()
