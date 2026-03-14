from ultralytics import YOLO

# Load your local PyTorch model
model = YOLO("./model.pt")

# Export to CoreML format
# half=True: Uses FP16 precision (optimized for Apple Neural Engine)
# nms=True: Includes Non-Maximum Suppression inside the model for easier use
model.export(format="coreml", half=True, nms=True)
