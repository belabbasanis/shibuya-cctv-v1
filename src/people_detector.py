import cv2
from ultralytics import YOLO
import numpy as np

class PeopleDetector:
    def __init__(self, model_path="yolov8m.pt"):
        # Load YOLO model
        try:
            self.model = YOLO(model_path)
            print(f"✅ YOLO model loaded successfully from {model_path}")
        except Exception as e:
            print(f"❌ Failed to load YOLO model: {e}")
            self.model = None

    def detect_people(self, frame):
        # Ensure the model is loaded
        if not self.model:
            raise ValueError("YOLO model is not loaded. Check the model file path.")

        # Run YOLO inference on the frame
        results = self.model(frame)

        # Process detections
        filtered_frame = frame.copy()  # Create a copy to draw bounding boxes
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                confidence = box.conf[0].item()  # Confidence score
                class_id = int(box.cls[0].item())  # Class ID (0 = person)

                if class_id == 0 and confidence > 0.5:  # Only detect people
                    cv2.rectangle(filtered_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue bounding box
                    cv2.putText(filtered_frame, f"Person ({confidence:.2f})", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        return filtered_frame  # Return the modified frame


if __name__ == "__main__":
    video_path = "input_videos/shibuya.mp4"
    cap = cv2.VideoCapture(video_path)
    detector = PeopleDetector(model_path="yolov8m.pt")  # Use yolo11.pt as the model file

    frame_index = 0  # Track frame count

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Skip processing some frames to speed up
        if frame_index % 2 == 0:  # Process every 2nd frame
            frame_resized = cv2.resize(frame, (640, 360))  # Resize for speed
            processed_frame = detector.detect_people(frame_resized)  # Call the detect_people method
            cv2.imshow("YOLO Detection (yolov8m.pt)", processed_frame)

        frame_index += 1  # Increment frame counter

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Release resources outside the loop
    cap.release()
    cv2.destroyAllWindows()
