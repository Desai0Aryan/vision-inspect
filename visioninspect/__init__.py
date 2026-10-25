"""VisionInspect: real-time computer-vision inspection system."""

__version__ = "0.1.0"

# Defect taxonomy shared by the data generator, models, API and UI.
CLASSES = ["scratch", "crack", "stain", "missing_part"]
CLASS_COLORS_BGR = {
    "scratch": (0, 200, 255),     # amber
    "crack": (60, 60, 255),       # red
    "stain": (255, 140, 40),      # blue
    "missing_part": (200, 60, 220),  # magenta
}
