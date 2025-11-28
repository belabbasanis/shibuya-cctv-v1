# Shibuya CCTV Detection System

A computer vision application for real-time people detection in video streams using YOLOv8. This project processes CCTV footage to detect and track people with bounding box annotations.

## Features

- 🎥 Real-time people detection in video streams
- 🔍 YOLOv8-based object detection
- 📊 Confidence score visualization
- 🎬 Batch video processing support
- 🖼️ Interactive video playback with detections

## Requirements

- Python 3.9+
- CUDA-capable GPU (optional, but recommended for faster inference)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/belabbasanis/shibuya-cctv-v1.git
cd shibuya-cctv-v1
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download YOLO model files:
   - The application uses YOLOv8 models (`.pt` files)
   - Models will be automatically downloaded on first use, or you can download them manually:
     - [YOLOv8n](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt) (nano - fastest)
     - [YOLOv8s](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s.pt) (small)
     - [YOLOv8m](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m.pt) (medium - recommended)
     - [YOLOv8l](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8l.pt) (large)
     - [YOLOv8x](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x.pt) (extra large - most accurate)

## Usage

### Interactive People Detector

Run the interactive detector to view video with real-time detections:

```bash
python src/people_detector.py
```

**Features:**
- Displays video with bounding boxes around detected people
- Shows confidence scores
- Press 'q' to quit
- Processes every 2nd frame for better performance

**Configuration:**
- Default video: `input_videos/shibuya.mp4`
- Default model: `yolov8m.pt`
- Confidence threshold: 0.5

### Batch Video Processing

Process a video file and save the output:

```bash
python src/video_detection.py
```

**Features:**
- Processes entire video
- Saves output as `shibuya_detected.mp4`
- Confidence threshold: 0.4

**Configuration:**
- Input video: `input_videos/shibuya.mp4`
- Output video: `shibuya_detected.mp4`
- Model: `yolov8m.pt`

### Using the PeopleDetector Class

You can also use the `PeopleDetector` class in your own code:

```python
from src.people_detector import PeopleDetector
import cv2

# Initialize detector
detector = PeopleDetector(model_path="yolov8m.pt")

# Process a frame
cap = cv2.VideoCapture("input_videos/shibuya.mp4")
ret, frame = cap.read()

if ret:
    processed_frame = detector.detect_people(frame)
    cv2.imshow("Detection", processed_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

cap.release()
```

## Project Structure

```
shibuya-cctv-v1/
├── src/
│   ├── people_detector.py      # PeopleDetector class and interactive script
│   ├── video_detection.py      # Batch video processing script
│   ├── main.py                 # Main entry point (placeholder)
│   ├── video_processor.py      # Video processing utilities (placeholder)
│   ├── utils.py                # Utility functions (placeholder)
│   ├── color_filter.py         # Color filtering utilities (placeholder)
│   └── export_yolo_to_onnx.py # Model export utilities
├── config/
│   └── settings.py            # Configuration settings
├── input_videos/              # Input video files (not tracked in git)
├── output/                    # Output directory (not tracked in git)
├── requirements.txt           # Python dependencies
├── .gitignore                 # Git ignore rules
└── README.md                  # This file
```

## Configuration

### Model Selection

You can use different YOLO models based on your needs:

- **YOLOv8n**: Fastest, lowest accuracy
- **YOLOv8s**: Balanced speed/accuracy
- **YOLOv8m**: Recommended for most use cases
- **YOLOv8l**: Higher accuracy, slower
- **YOLOv8x**: Highest accuracy, slowest

Change the model by modifying the `model_path` parameter:

```python
detector = PeopleDetector(model_path="yolov8n.pt")  # Use nano model
```

### Confidence Threshold

Adjust the confidence threshold to filter detections:

- In `people_detector.py`: Line 31, change `confidence > 0.5`
- In `video_detection.py`: Line 28, change `conf=0.4`

Lower values = more detections (including false positives)
Higher values = fewer detections (more accurate)

## Troubleshooting

### Model Not Found
If you get a model loading error:
- Ensure the model file (`.pt`) exists in the project root
- Models are automatically downloaded on first use
- Check your internet connection for automatic downloads

### Video File Not Found
- Ensure video files are in the `input_videos/` directory
- Check the file path in the script matches your video location

### OpenCV Display Issues
- If running on a headless server, the interactive mode won't work
- Use `video_detection.py` for batch processing instead
- For headless systems, consider using Xvfb or similar

### Performance Issues
- Use smaller models (YOLOv8n or YOLOv8s) for faster processing
- Reduce video resolution in `people_detector.py` (line 53)
- Process fewer frames (increase the skip rate on line 52)

## Dependencies

Key dependencies:
- `ultralytics` - YOLOv8 implementation
- `opencv-python` - Video processing and display
- `torch` - Deep learning framework
- `numpy` - Numerical operations

See `requirements.txt` for the complete list.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

See [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLOv8
- OpenCV community for computer vision tools

## Future Improvements

- [ ] Add tracking capabilities (track individuals across frames)
- [ ] Support for multiple object classes
- [ ] Real-time streaming from cameras
- [ ] Web interface for video upload and processing
- [ ] Database integration for detection logging
- [ ] Improved error handling and resource management
- [ ] Configuration file support
- [ ] Multi-threaded processing for better performance

