from ultralytics import YOLO
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import platform
import os
import sys
from pathlib import Path

# Add config directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import styling

# Load YOLOv8n model (nano - fastest, good for real-time processing)
model = YOLO("yolov8n.pt")

# Load font from styling config
def load_font():
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

inter_font = load_font()

def draw_text_with_font(frame, text, position, bg_color, text_color, font):
    """Draw text with configured font using PIL, then composite onto OpenCV frame"""
    if not styling.SHOW_LABEL:
        return frame
        
    x, y = position
    try:
        # Convert OpenCV frame to PIL Image
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)
        
        # Get text bounding box
        bbox = draw.textbbox((0, 0), text, font=font)
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
        draw.text((x, y - text_height), text, fill=text_color, font=font)
        
        # Convert back to OpenCV format
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    except Exception as e:
        # Fallback to OpenCV text if PIL fails
        cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
        return frame

# Open video file
video_path = "input_videos/cctv-subway-nyc.mp4"  # Change this to your video file
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for MP4 output

# Create video writer for output
output_path = "shibuya_detected.mp4"
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Process each frame with proper resource management
try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Exit loop if video ends

        # Run YOLOv8n on frame with configured confidence threshold
        results = model(frame, conf=styling.CONFIDENCE_THRESHOLD)

        # Draw detections on frame with yellow boxes and labels
        for result in results:
            boxes = result.boxes
            if hasattr(boxes, 'xyxy') and boxes.xyxy is not None:
                for i, box in enumerate(boxes.xyxy):
                    x1, y1, x2, y2 = map(int, box[:4])  # Bounding box coordinates
                    confidence = boxes.conf[i].item() if hasattr(boxes, 'conf') and i < len(boxes.conf) else 0.0
                    class_id = int(boxes.cls[i].item()) if hasattr(boxes, 'cls') and i < len(boxes.cls) else -1
                    
                    if class_id == 0:  # Only draw people
                        # Draw bounding box if enabled
                        if styling.SHOW_BOX:
                            cv2.rectangle(frame, (x1, y1), (x2, y2), 
                                         styling.BOX_COLOR_BGR, styling.BOX_THICKNESS)
                        
                        # Draw label if enabled
                        if styling.SHOW_LABEL:
                            # Format label using config format string
                            if styling.SHOW_CONFIDENCE:
                                label = styling.LABEL_FORMAT.format(percentage=int(confidence * 100))
                            else:
                                label = styling.LABEL_FORMAT.replace(" {percentage}%", "")
                            
                            # Draw label with configured styling
                            frame = draw_text_with_font(
                                frame, label, 
                                (x1, y1 + styling.LABEL_OFFSET_Y),
                                styling.LABEL_BACKGROUND_COLOR_RGB,
                                styling.LABEL_TEXT_COLOR_RGB,
                                inter_font
                            )

        # Write processed frame to output video
        out.write(frame)
finally:
    # Always release resources, even if an exception occurs
    cap.release()
    out.release()
    cv2.destroyAllWindows()

print(f"Processed video saved as: {output_path}")