import cv2
import torch
import numpy as np
from torchvision import models, transforms

# Load DeepLabV3 with ResNet-101 backbone (optimized for speed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.segmentation.deeplabv3_resnet101(pretrained=True).to(device).eval()

# Image Preprocessing (Reduce input size for faster inference)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),  # Reduce input size for speed
    transforms.ToTensor(),
])

cap = cv2.VideoCapture(0)  # Open Webcam
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % 2 == 0:  # Skip every 2nd frame to boost FPS
        continue

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_tensor = transform(img).unsqueeze(0).to(device)  # Move to GPU if available

    # Run Model Inference
    with torch.no_grad():
        output = model(img_tensor)['out'][0].argmax(0).byte().cpu().numpy()

    # Resize segmentation map to match frame size (fastest interpolation)
    seg_map = cv2.resize(output, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Convert to BGR if needed
    if len(seg_map.shape) == 2:
        seg_map = cv2.cvtColor(seg_map, cv2.COLOR_GRAY2BGR)

    # Apply Color Map (avoids costly blending)
    seg_map = cv2.applyColorMap(seg_map * 10, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(frame, 0.5, seg_map, 0.5, 0)

    # Display only the segmentation mask for speed boost
    cv2.imshow("Segmentation", overlay)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
