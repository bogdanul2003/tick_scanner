import time
import glob
import os
from ultralytics import YOLO

# Load the newly created CoreML package
coreml_model = YOLO("./model.mlpackage")

# Get list of images
image_files = glob.glob("pics/*.png") + glob.glob("pics/*.jpg") + glob.glob("pics/*.jpeg")

if not image_files:
    print("No images found in pics/ folder")
    exit()

# Warmup run (to load shaders/allocate memory)
print("Warming up...")
coreml_model.predict(source=image_files[0], verbose=False)

# Run prediction - it will automatically use the M4's hardware accelerators
print(f"Processing {len(image_files)} images...")
total_start = time.time()

for img_path in image_files:
    start_time = time.time()
    results = coreml_model.predict(source=img_path, save=True)
    end_time = time.time()
    print(f"{os.path.basename(img_path)}: {end_time - start_time:.4f} seconds")

total_end = time.time()
print(f"Total time: {total_end - total_start:.4f} seconds")
