from ultralytics import YOLO
import cv2

# Load YOLOv8m model (better accuracy than YOLOv8n)
model = YOLO("yolov8m.pt")

# Open video file
video_path = "input_videos/shibuya.mp4"  # Change this to your video file
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for MP4 output

# Create video writer for output
output_path = "shibuya_detected.mp4"
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Process each frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Exit loop if video ends

    # Run YOLOv8m on frame
    results = model(frame, conf=0.4)  # Adjust confidence threshold as needed

    # Draw detections on frame
    for result in results:
        for box in result.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box[:4])  # Bounding box coordinates
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Write processed frame to output video
    out.write(frame)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Processed video saved as: {output_path}")