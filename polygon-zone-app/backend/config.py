# backend/config.py

import numpy as np
import os

# Get the directory where the config.py file is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Server Configuration
HTTP_HOST = "localhost"
HTTP_PORT = 8000 # Use a different port for HTTP uploads
WS_HOST = "localhost"
WS_PORT = 8765

# Upload and Output Configuration
# Use fixed folders relative to the backend directory
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "outputs")
MAX_FILE_SIZE_MB = 500 # Maximum file size for uploads

# CORS Configuration (Adjust origins as needed for your frontend deployment)
ALLOWED_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"] # Example for Vite dev server, added 127.0.0.1

# YOLOv8 Class Names (matching your original script)
CLASS_NAMES = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorbike', 4: 'aeroplane', 5: 'bus', 6: 'train', 7: 'truck',
    8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
    14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear',
    22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase',
    29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
    35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
    40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana',
    47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza',
    54: 'donut', 55: 'cake', 56: 'chair', 57: 'sofa', 58: 'pottedplant', 59: 'bed', 60: 'diningtable',
    61: 'toilet', 62: 'tvmonitor', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard',
    67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator',
    73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier',
    79: 'toothbrush'
}

# Frontend class names to Backend integer IDs mapping (for lookup)
FRONTEND_CLASS_TO_BACKEND_ID = {v: k for k, v in CLASS_NAMES.items()}

# Backend integer IDs to Frontend class names mapping (for results)
BACKEND_ID_TO_FRONTEND_CLASS = {k: v for k, v in CLASS_NAMES.items()}