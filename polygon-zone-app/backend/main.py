# backend/main.py

import argparse
import asyncio
from ultralytics import YOLO
import os # Import os for path validation and folder creation
import traceback # Import traceback

from server import start_servers
from config import HTTP_HOST, HTTP_PORT, WS_HOST, WS_PORT, UPLOAD_FOLDER, OUTPUT_FOLDER # Import config for reference and folder paths


def parse_arguments():
    parser = argparse.ArgumentParser(description="YOLOv8 Object Tracker Backend Server (Upload & WS)")
    parser.add_argument("--model", required=False, help="Path to the YOLOv8 model (.pt file)", default = "yolov8m.pt")

    args = parser.parse_args()

    # Basic check if model file exists
    if not os.path.exists(args.model):
        print(f"Error: Model file not found at {args.model}")
        exit(1) # Exit with error code

    return args

def main():
    args = parse_arguments()

    # --- Create necessary folders on startup ---
    try:
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        print(f"Ensured upload folder exists: {UPLOAD_FOLDER}")
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        print(f"Ensured output folder exists: {OUTPUT_FOLDER}")
    except Exception as e:
        print(f"Error creating necessary folders: {e}")
        traceback.print_exc()
        exit(1)

    # Load the YOLO model once at startup
    try:
        print(f"Loading model from {args.model}...")
        # You might want to add device selection here, e.g., model = YOLO(args.model).to('cuda')
        model = YOLO(args.model)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        traceback.print_exc()
        exit(1) # Exit with error code


    # Start both the HTTP upload server and the WebSocket server
    try:
        asyncio.run(start_servers(model))
    except KeyboardInterrupt:
        print("\nServers stopped by user.")
    except Exception as e:
        print(f"An unhandled error occurred while running the servers: {e}")
        traceback.print_exc()
        exit(1) # Exit with error code


if __name__ == "__main__":
    main()