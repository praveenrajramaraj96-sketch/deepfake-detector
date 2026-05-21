from transformers import pipeline
from PIL import Image
import torch

try:
    device_idx = 0 if torch.cuda.is_available() else -1
    deepfake_pipeline = pipeline("image-classification", model="prithivMLmods/Deep-Fake-Detector-v2-Model", device=device_idx)

    img2 = Image.open('tampered.jpg')
    print(deepfake_pipeline(img2))
except Exception as e:
    print(e)
