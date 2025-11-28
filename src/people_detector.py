import cv2
from ultralytics import YOLO
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import platform
import os
import sys
from pathlib import Path

# Add config directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import styling

class PeopleDetector:
    def __init__(self, model_path="yolov8n.pt"):
        # Load YOLO model
        try:
            self.model = YOLO(model_path)
            print(f"✅ YOLO model loaded successfully from {model_path}")
        except Exception as e:
            print(f"❌ Failed to load YOLO model: {e}")
            self.model = None
        
        # Load font from styling config
        self.font = self._load_font()
    
    def _load_font(self):
        """Load font from styling configuration"""
        try:
            system = platform.system()
            font_paths = styling.FONT_PATHS.get(system, [])
            
            for path in font_paths:
                try:
                    expanded_path = os.path.expanduser(path)
                    if os.path.exists(expanded_path):
                        font = ImageFont.truetype(expanded_path, styling.LABEL_FONT_SIZE)
                        print("✅ Inter font loaded successfully")
                        return font
                except:
                    continue
            
            # Fallback to default font
            font = ImageFont.load_default()
            print("⚠️ Inter font not found, using default font")
            return font
        except Exception as e:
            font = ImageFont.load_default()
            print(f"⚠️ Could not load Inter font: {e}, using default font")
            return font
    
    def _draw_text_with_font(self, frame, text, position, bg_color, text_color):
        """Draw text with configured font using PIL, then composite onto OpenCV frame"""
        if not styling.SHOW_LABEL:
            return frame
            
        x, y = position
        try:
            # Convert OpenCV frame to PIL Image
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_image)
            
            # Get text bounding box
            bbox = draw.textbbox((0, 0), text, font=self.font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # Draw background rectangle using config padding
            bg_coords = [
                x - styling.LABEL_PADDING,
                y - text_height - styling.LABEL_PADDING,
                x + text_width + styling.LABEL_PADDING,
                y + styling.LABEL_PADDING
            ]
            draw.rectangle(bg_coords, fill=bg_color)
            
            # Draw text
            draw.text((x, y - text_height), text, fill=text_color, font=self.font)
            
            # Convert back to OpenCV format
            frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            return frame
        except Exception as e:
            # Fallback to OpenCV text if PIL fails
            cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
            return frame

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

                if class_id == 0 and confidence >= styling.CONFIDENCE_THRESHOLD:
                    # Draw bounding box if enabled
                    if styling.SHOW_BOX:
                        cv2.rectangle(filtered_frame, (x1, y1), (x2, y2), 
                                     styling.BOX_COLOR_BGR, styling.BOX_THICKNESS)
                    
                    # Draw label if enabled
                    if styling.SHOW_LABEL:
                        # Format label using config format string
                        if styling.SHOW_CONFIDENCE:
                            label = styling.LABEL_FORMAT.format(percentage=int(confidence * 100))
                        else:
                            label = styling.LABEL_FORMAT.replace(" {percentage}%", "")
                        
                        # Draw label with configured styling
                        filtered_frame = self._draw_text_with_font(
                            filtered_frame, label, 
                            (x1, y1 + styling.LABEL_OFFSET_Y),
                            styling.LABEL_BACKGROUND_COLOR_RGB,
                            styling.LABEL_TEXT_COLOR_RGB
                        )

        return filtered_frame  # Return the modified frame


if __name__ == "__main__":
    video_path = "input_videos/cctv-subway-nyc.mp4"
    cap = cv2.VideoCapture(video_path)
    detector = PeopleDetector(model_path="yolov8n.pt")  # Using nano model for faster inference

    frame_index = 0  # Track frame count

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Skip processing some frames to speed up
            if frame_index % styling.FRAME_SKIP == 0:
                # Resize for speed if configured
                if styling.RESIZE_WIDTH and styling.RESIZE_HEIGHT:
                    frame_resized = cv2.resize(frame, (styling.RESIZE_WIDTH, styling.RESIZE_HEIGHT))
                else:
                    frame_resized = frame
                processed_frame = detector.detect_people(frame_resized)  # Call the detect_people method
                cv2.imshow("YOLO Detection (yolov8n.pt)", processed_frame)

            frame_index += 1  # Increment frame counter

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
    finally:
        # Always release resources, even if an exception occurs
        cap.release()
        cv2.destroyAllWindows()
