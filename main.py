import cv2
import torch
import numpy as np
from torchvision import models, transforms

# Load Pretrained DeepLabV3 Model
model = models.segmentation.deeplabv3_resnet101(pretrained=True).eval()

# Image Preprocessing Function
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])

# Open Webcam
cap = cv2.VideoCapture(0)  # 0 = Default Webcam

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    img_tensor = transform(img).unsqueeze(0)  # Convert to Tensor

    # Run Segmentation
    with torch.no_grad():
        output = model(img_tensor)['out'][0]
        output = torch.argmax(output, dim=0).byte().cpu().numpy()

    # Resize segmentation map to match frame size
    seg_map = cv2.resize(output, (frame.shape[1], frame.shape[0]))

    # Convert to BGR if needed
    if len(seg_map.shape) == 2:
        seg_map = cv2.cvtColor(seg_map, cv2.COLOR_GRAY2BGR)

    # Apply Color Map for better visualization
    seg_map = cv2.applyColorMap(seg_map * 10, cv2.COLORMAP_JET)

    # Blend images
    overlay = cv2.addWeighted(frame, 0.5, seg_map, 0.5, 0)

    # Show Output
    cv2.imshow("Segmentation", overlay)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
