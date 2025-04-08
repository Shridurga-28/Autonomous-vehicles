import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch

from bisenetv2 import BiSeNet  # Import your trained model

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model architecture
model = BiSeNet(num_classes=256).to(device)

# Load trained weights
model.load_state_dict(torch.load("bisenet_model.pth", map_location=device), strict=False)

model.eval()

# Define image transformation
transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

def predict(frame):
    """Predict segmentation mask for a given video frame."""
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
    image = transform(image=image)["image"].unsqueeze(0).to(device)  # Apply transformations

    with torch.no_grad():
        output = model(image).argmax(dim=1).cpu().numpy()[0]  # Get segmentation mask

    return output

def segment_video(input_video_path, output_video_path):
    """Process a video frame by frame and save segmented output."""
    cap = cv2.VideoCapture(input_video_path)
    
    if not cap.isOpened():
        print("Error: Cannot open video file")
        return
    
    # Get video properties
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(5))  # Frames per second

    # Define video writer for saving output
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    print("Processing video...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        # Get segmentation mask
        segmentation_map = predict(frame)

        # Resize mask back to original frame size
        mask_resized = cv2.resize(segmentation_map, (frame_width, frame_height), interpolation=cv2.INTER_NEAREST)

        # Apply color map for visualization
        mask_colored = cv2.applyColorMap((mask_resized * 10).astype(np.uint8), cv2.COLORMAP_JET)

        # Overlay segmentation on original frame
        overlayed_frame = cv2.addWeighted(frame, 0.6, mask_colored, 0.4, 0)

        # Save frame to output video
        out.write(overlayed_frame)

    cap.release()
    out.release()
    print("Processing complete. Saved output to:", output_video_path)

# Define input and output video paths
input_video = r"C:\Users\SRI SAIRAM COLLEGE\Documents\UNET\v3.mp4"  # Change to your video path
output_video = "segmented_output3.avi"

# Process video and save output
segment_video(input_video, output_video)