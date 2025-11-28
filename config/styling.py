"""
Global styling configuration for detection bounding boxes and labels.
Edit these values to customize the appearance of detections across the application.
Colors are specified in hex format (e.g., "#FFFF00" for yellow, "#000000" for black).
"""

# Color conversion utilities
def hex_to_rgb(hex_color):
    """Convert hex color (#RRGGBB) to RGB tuple (R, G, B)"""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def hex_to_bgr(hex_color):
    """Convert hex color (#RRGGBB) to BGR tuple (B, G, R) for OpenCV"""
    rgb = hex_to_rgb(hex_color)
    return (rgb[2], rgb[1], rgb[0])  # Reverse RGB to BGR

# Bounding Box Styling
BOX_COLOR_HEX = "#F3BE0C"  # Yellow in hex format
BOX_THICKNESS = 2  # Thickness of bounding box lines

# Label Styling
LABEL_BACKGROUND_COLOR_HEX = "#F3BE0C"  # Yellow background for labels (hex)
LABEL_TEXT_COLOR_HEX = "#000000"  # Black text (hex)
LABEL_PADDING = 4  # Padding around text in pixels
LABEL_FONT_SIZE = 18  # Font size in points

# Computed RGB/BGR values (automatically converted from hex)
BOX_COLOR_BGR = hex_to_bgr(BOX_COLOR_HEX)
BOX_COLOR_RGB = hex_to_rgb(BOX_COLOR_HEX)
LABEL_BACKGROUND_COLOR_RGB = hex_to_rgb(LABEL_BACKGROUND_COLOR_HEX)
LABEL_TEXT_COLOR_RGB = hex_to_rgb(LABEL_TEXT_COLOR_HEX)

# Label Position
LABEL_OFFSET_Y = -5  # Vertical offset from top of bounding box (negative = above box)

# Font Configuration
# Font paths to search for Inter font (will try in order)
FONT_PATHS = {
    "Darwin": [  # macOS
        "/System/Library/Fonts/Supplemental/Inter.ttc",
        "/Library/Fonts/Inter-Regular.ttf",
        "~/Library/Fonts/Inter-Regular.ttf",
        "~/Library/Fonts/Inter.ttf",
    ],
    "Windows": [
        "C:/Windows/Fonts/inter.ttf",
        "C:/Windows/Fonts/Inter-Regular.ttf",
        "C:/Windows/Fonts/Inter-Bold.ttf",
    ],
    "Linux": [
        "/usr/share/fonts/truetype/inter/Inter-Regular.ttf",
        "/usr/share/fonts/opentype/inter/Inter-Regular.ttf",
        "~/.fonts/Inter-Regular.ttf",
        "/usr/local/share/fonts/Inter-Regular.ttf",
    ]
}

# Detection Settings
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence to show detection (0.0 to 1.0)
LABEL_FORMAT = "Person {percentage}%"  # Format string for labels ({percentage} will be replaced)

# Advanced Styling Options
SHOW_CONFIDENCE = True  # Whether to show confidence percentage in label
SHOW_BOX = True  # Whether to draw bounding boxes
SHOW_LABEL = True  # Whether to show labels

# Performance Settings (for reference)
FRAME_SKIP = 2  # Process every Nth frame (1 = all frames, 2 = every other frame)
RESIZE_WIDTH = 640  # Resize frame width for processing (None to disable)
RESIZE_HEIGHT = 360  # Resize frame height for processing (None to disable)

