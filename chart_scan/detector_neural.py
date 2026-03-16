import time
import glob
import os
import argparse
from ultralytics import YOLO

# Parse command line arguments
parser = argparse.ArgumentParser(description="Run YOLO inference on images")
parser.add_argument("--show-boxes", action="store_true", help="Print detected boxes for each image")
parser.add_argument("--find-x", type=int, default=510, help="Find images with boxes extending past this x coordinate")
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
images_with_x_past_threshold = []

# Define custom save folder for filtered images
results_folder = "runs/detect/filtered_results"
os.makedirs(results_folder, exist_ok=True)

for img_path in image_files:
    start_time = time.time()
    # Don't use save=True globally
    results = coreml_model.predict(source=img_path, save=False, verbose=False)
    
    # Check if any box extends past find_x
    has_box_past_threshold = False
    
    if results[0].boxes is not None:
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy.tolist()[0]
            
            # Check if x2 is past the threshold
            if x2 > args.find_x:
                has_box_past_threshold = True
                break

    # If threshold is exceeded, save the annotated image
    if has_box_past_threshold:
        images_with_x_past_threshold.append(os.path.basename(img_path))
        # Save to the specific filtered_results folder
        # We manually save the plotted image
        annotated_img = results[0].plot() # returns a numpy array
        import cv2
        save_path = os.path.join(results_folder, os.path.basename(img_path))
        cv2.imwrite(save_path, annotated_img)
        print(f"  [MATCH] Saved to {save_path}")
    
    # Print detected boxes if flag is set
    if args.show_boxes and results[0].boxes is not None:
        print(f"\nDetections in {os.path.basename(img_path)}:")
        for box in results[0].boxes:
            class_id = int(box.cls.item())
            class_name = coreml_model.names[class_id]
            print(f"  {class_name}: {box.xyxy.tolist()} | Confidence: {box.conf.item():.4f}")
    
    end_time = time.time()
    print(f"{os.path.basename(img_path)}: {end_time - start_time:.4f} seconds")

total_end = time.time()
print(f"Total time: {total_end - total_start:.4f} seconds")

# Print images with boxes extending past threshold
print(f"\nImages with boxes extending past x={args.find_x}:")
if images_with_x_past_threshold:
    for img_name in images_with_x_past_threshold:
        print(f"  {img_name}")
else:
    print("  None found")
