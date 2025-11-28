from ultralytics import YOLO

# Load YOLOv8 Nano model (smallest version)
model = YOLO("yolov8n.pt")

# Convert to ONNX format
model.export(format="onnx")

print("✅ YOLOv8 model exported to 'yolov8n.onnx'")
