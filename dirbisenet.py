import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch

from bisenetv2 import BiSeNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model = BiSeNet(num_classes=256).to(device)
model.load_state_dict(torch.load("bisenet_model.pth", map_location=device), strict=False)
model.eval()

transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

def predict(frame):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = transform(image=image)["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image).argmax(dim=1).cpu().numpy()[0]

    return output

def get_lane_direction(mask):
    """
    Fit a polynomial to the lane and estimate curve direction.
    """
    height, width = mask.shape
    lane_points = np.column_stack(np.where(mask > 0))  # (y, x)

    if len(lane_points) < 500:
        return "No lane detected"

    # Reverse (y, x) -> (x, y) and normalize
    x = lane_points[:, 1]
    y = lane_points[:, 0]

    # Fit a 2nd-degree polynomial: y = Ax^2 + Bx + C
    poly = np.polyfit(y, x, 2)

    # Radius of curvature formula
    y_eval = np.max(y)
    A = poly[0]
    B = poly[1]

    radius = ((1 + (2*A*y_eval + B)**2)**1.5) / np.abs(2*A)

    # Determine direction
    if radius > 3000:
        return "Straight"
    elif A > 0:
        return "Right Turn"
    else:
        return "Left Turn"

def segment_video(input_video_path, output_video_path):
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Cannot open video file")
        return
    
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(5))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    print("Processing video...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        segmentation_map = predict(frame)
        mask_resized = cv2.resize(segmentation_map, (frame_width, frame_height), interpolation=cv2.INTER_NEAREST)

        # Calculate direction
        direction = get_lane_direction(mask_resized)

        # Apply colormap
        mask_colored = cv2.applyColorMap((mask_resized * 10).astype(np.uint8), cv2.COLORMAP_JET)
        overlayed_frame = cv2.addWeighted(frame, 0.6, mask_colored, 0.4, 0)

        # Put direction text on the frame
        cv2.putText(overlayed_frame, f"Direction: {direction}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 255, 0), 2, cv2.LINE_AA)

        out.write(overlayed_frame)

    cap.release()
    out.release()
    print("Processing complete. Saved output to:", output_video_path)

# Run
input_video = r"C:\Users\SRI SAIRAM COLLEGE\Documents\UNET\extra.mp4"
output_video = "segmented_output_1.avi"
segment_video(input_video, output_video)
