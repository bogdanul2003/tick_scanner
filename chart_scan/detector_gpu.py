import time
import glob
import os
from ultralytics import YOLO
import torch

# 1. Verify your Mac's GPU is accessible to PyTorch
if torch.backends.mps.is_available():
    print("M4 GPU (MPS) is available!")
else:
    print("MPS not found, falling back to CPU.")

# 2. Load your local .pt model
model = YOLO("model.pt")

# Get list of images
image_files = glob.glob("pics/*.png") + glob.glob("pics/*.jpg") + glob.glob("pics/*.jpeg")

if not image_files:
    print("No images found in pics/ folder")
    exit()

# Warmup run (to load shaders/allocate memory)
print("Warming up...")
model.predict(source=image_files[0], device='mps', verbose=False)

# 3. Explicitly target the GPU with device='mps'
print(f"Processing {len(image_files)} images...")
total_start = time.time()

for img_path in image_files:
    start_time = time.time()
    results = model.predict(source=img_path, device='mps', save=True)
    end_time = time.time()
    print(f"{os.path.basename(img_path)}: {end_time - start_time:.4f} seconds")

total_end = time.time()
print(f"Total time: {total_end - total_start:.4f} seconds")
