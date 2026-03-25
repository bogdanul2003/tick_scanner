import time
import glob
import os
import argparse
from ultralytics import YOLO
import cv2

def run_detection(input_dir, output_dir, find_x=510, show_boxes=False):
    # Load the CoreML package
    # Path is relative to the root if called from src/api.py, 
    # but let's make it more robust.
    model_path = os.path.join(os.path.dirname(__file__), "model.mlpackage")
    coreml_model = YOLO(model_path)

    # Get list of images
    image_files = glob.glob(os.path.join(input_dir, "*.png")) + \
                  glob.glob(os.path.join(input_dir, "*.jpg")) + \
                  glob.glob(os.path.join(input_dir, "*.jpeg"))

    if not image_files:
        print(f"No images found in {input_dir}")
        return []

    # Warmup run
    print("Warming up model...")
    coreml_model.predict(source=image_files[0], verbose=False, conf=0.3)

    print(f"Processing {len(image_files)} images from {input_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    
    total_start = time.time()
    images_with_x_past_threshold = []

    for img_path in image_files:
        start_time = time.time()
        results = coreml_model.predict(source=img_path, save=False, verbose=False, conf=0.3)
        
        has_box_past_threshold = False
        rightmost_pattern = None
        max_x2 = -1

        if results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy.tolist()[0]
                
                # Update rightmost pattern if this box is further right
                if x2 > max_x2:
                    max_x2 = x2
                    class_id = int(box.cls.item())
                    rightmost_pattern = coreml_model.names[class_id]

                if x2 > find_x:
                    has_box_past_threshold = True

        if has_box_past_threshold:
            filename = os.path.basename(img_path)
            
            # Extract all boxes for this image to return them
            detected_boxes = []
            if results[0].boxes is not None:
                for box in results[0].boxes:
                    coords = box.xyxy.tolist()[0] # [x1, y1, x2, y2]
                    detected_boxes.append({
                        "box": coords,
                        "conf": float(box.conf.item()),
                        "class": int(box.cls.item()),
                        "name": coreml_model.names[int(box.cls.item())]
                    })

            images_with_x_past_threshold.append({
                "filename": filename,
                "rightmost_pattern": rightmost_pattern,
                "boxes": detected_boxes
            })
            
            # Save the annotated image
            annotated_img = results[0].plot()
            save_path = os.path.join(output_dir, filename)
            cv2.imwrite(save_path, annotated_img)
            print(f"  [MATCH] Saved to {save_path} (Rightmost: {rightmost_pattern})")
        
        if show_boxes and results[0].boxes is not None:
            print(f"\nDetections in {os.path.basename(img_path)}:")
            for box in results[0].boxes:
                class_id = int(box.cls.item())
                class_name = coreml_model.names[class_id]
                print(f"  {class_name}: {box.xyxy.tolist()} | Confidence: {box.conf.item():.4f}")
        
        end_time = time.time()
        # print(f"{os.path.basename(img_path)}: {end_time - start_time:.4f} seconds")

    total_end = time.time()
    print(f"Detection completed in {total_end - total_start:.4f} seconds. Found {len(images_with_x_past_threshold)} matches.")
    return images_with_x_past_threshold

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run YOLO inference on images")
    parser.add_argument("--input-dir", type=str, default="pics", help="Folder containing input images")
    parser.add_argument("--output-dir", type=str, default="runs/detect/filtered_results", help="Folder to save filtered images")
    parser.add_argument("--show-boxes", action="store_true", help="Print detected boxes for each image")
    parser.add_argument("--find-x", type=int, default=510, help="Find images with boxes extending past this x coordinate")
    args = parser.parse_args()

    run_detection(args.input_dir, args.output_dir, args.find_x, args.show_boxes)
