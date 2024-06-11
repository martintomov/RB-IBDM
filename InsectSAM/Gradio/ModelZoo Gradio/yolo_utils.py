from ultralytics import YOLO
import torch
import numpy as np
import cv2
from huggingface_hub import hf_hub_download

REPO_ID = "idml/Yolov8_InsectDetect"
FILENAME = "insectYolo.pt"


# Ensure you have the model file
model = YOLO(hf_hub_download(repo_id=REPO_ID, filename=FILENAME))
def yolo_processimage(image):
    results = model(source=image,
                conf=0.2, device='cpu')
    rgb_image = cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB)
    return rgb_image

