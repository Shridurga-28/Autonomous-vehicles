import cv2
import numpy as np

mask_path = r"C:\Users\SRI SAIRAM COLLEGE\Documents\UNET\training\labels\__CRyFzoDOXn6unQ6a3DnQ.png"

mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

unique_classes = np.unique(mask)
print("Unique class labels in the mask:", unique_classes)

