import time
import glob
import os
import argparse
from ultralytics import YOLO

# Parse command line arguments
parser = argparse.ArgumentParser(description="Run YOLO inference on images")
parser.add_argument("--show-boxes", action="store_true", help="Print detected boxes for each image")
args = parser.parse_args()

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
    
    # Print detected boxes if flag is set
    if args.show_boxes:
        print(f"\nDetections in {os.path.basename(img_path)}:")
        if results[0].boxes is not None:
            for box in results[0].boxes:
                class_id = int(box.cls.item())
                class_name = coreml_model.names[class_id]
                print(f"  {class_name}: {box.xyxy.tolist()} | Confidence: {box.conf.item():.4f}")
        else:
            print("  No boxes detected")
    
    end_time = time.time()
    print(f"{os.path.basename(img_path)}: {end_time - start_time:.4f} seconds")

total_end = time.time()
print(f"Total time: {total_end - total_start:.4f} seconds")
