import cv2
import json
import numpy as np
from datetime import datetime
from roboflow import Roboflow
import ollama


roboflow_api_key = "Your API key here"
# Initialize Roboflow model
rf = Roboflow(api_key=roboflow_api_key)
project = rf.workspace().project("parking_lot-hnmz5")
model = project.version("3").model

# Initialize report data storage
report_data = []

# Function to generate LLM summary with Ollama
def summarize_parking_status(occupied, available, timestamp):
    prompt = f"""
    Analyze the parking lot data below:
    - Timestamp: {timestamp}
    - Occupied Spaces: {occupied}
    - Available Spaces: {available}

    Provide detailed analysis, potential risks, and recommendations for improving space management.
    """
    response = ollama.chat(
        model="phi4", ##change the model here
        messages=[{"role": "user", "content": prompt}]
    )
    return response['message']['content']

# Function to save data to Markdown
def save_report_to_md(report_data, filename="Parking_Report.md"):
    with open(filename, "w", encoding="utf-8") as f:
        f.write("# 🚗 Parking Lot Analysis Report\n\n")
        f.write(f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## 📊 Occupancy Summary Table\n\n")
        f.write("| Timestamp           | Occupied Spaces | Available Spaces |\n")
        f.write("|--------------------|-----------------|-----------------|\n")
        for entry in report_data:
            f.write(f"| {entry['timestamp']} | {entry['occupied']} | {entry['available']} |\n")

        f.write("\n## 📄 Detailed Analysis by LLM\n\n")
        for entry in report_data:
            f.write(f"### ⏰ Report at {entry['timestamp']}\n")
            f.write(f"- **Occupied Spaces:** {entry['occupied']}\n")
            f.write(f"- **Available Spaces:** {entry['available']}\n\n")
            f.write(f"> {entry['summary']}\n\n")
            f.write("---\n\n")

# Function to scale bounding box coordinates
def scale_bounding_boxes(json_file, original_size, target_size):
    with open(json_file, "r") as f:
        data = json.load(f)
    scale_x = target_size[0] / original_size[0]
    scale_y = target_size[1] / original_size[1]
    for polygon in data:
        for point in polygon["points"]:
            point[0] = int(point[0] * scale_x)
            point[1] = int(point[1] * scale_y)
    return data

# Add a background rectangle behind text
def draw_label_with_background(frame, text, position, bg_color, text_color):
    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    x, y = position
    cv2.rectangle(frame, (x - 10, y - text_height - 10), (x + text_width + 10, y + 10), bg_color, -1)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)

# Video setup
cap = cv2.VideoCapture("parking.mp4")
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
video_writer = cv2.VideoWriter("parking_analysis_with_styled_report.avi", cv2.VideoWriter_fourcc(*"XVID"), fps, (w, h))

# Scale bounding boxes
original_image_size = (2275, 1212)
video_size = (w, h)
scaled_data = scale_bounding_boxes("bounding_boxes.json", original_image_size, video_size)

# Updated Colors for Modern Look
occupied_color = (0, 0, 255)  # Bright Red
available_color = (0, 255, 0)  # Vibrant Green
glow_color = (255, 255, 255)  # Glow effect for style

# Process video frames
frame_counter = 0
summary_interval = 300  # Generate summary every 300 frames

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference on the frame
    response = model.predict(frame, confidence=40, overlap=30)
    predictions = response.json()['predictions']

    occupied_count = 0
    for region in scaled_data:
        points = np.array(region["points"], dtype=np.int32).reshape((-1, 1, 2))
        region_occupied = False

        for pred in predictions:
            x_min = int(pred['x'] - pred['width'] / 2)
            y_min = int(pred['y'] - pred['height'] / 2)
            x_max = int(pred['x'] + pred['width'] / 2)
            y_max = int(pred['y'] + pred['height'] / 2)
            bbox_center = ((x_min + x_max) // 2, (y_min + y_max) // 2)

            if cv2.pointPolygonTest(points, bbox_center, False) >= 0:
                region_occupied = True
                break

        color = occupied_color if region_occupied else available_color
        cv2.polylines(frame, [points], isClosed=True, color=glow_color, thickness=4)  # Glow effect
        cv2.polylines(frame, [points], isClosed=True, color=color, thickness=2)      # Main color
        occupied_count += region_occupied

    available_count = len(scaled_data) - occupied_count

    # Annotate frame with modern text overlay
    # With this (Plain Text):
    draw_label_with_background(frame, f"Occupied: {occupied_count}", (10, 40), (50, 50, 50, 100), occupied_color)
    draw_label_with_background(frame, f"Available: {available_count}", (10, 80), (50, 50, 50, 100), available_color)

    # Generate LLM Summary at intervals
    if frame_counter % summary_interval == 0:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        llm_summary = summarize_parking_status(occupied_count, available_count, timestamp)
        report_data.append({
            "timestamp": timestamp,
            "occupied": occupied_count,
            "available": available_count,
            "summary": llm_summary
        })

    video_writer.write(frame)
    frame_counter += 1

# Save report
save_report_to_md(report_data)

cap.release()
video_writer.release()
cv2.destroyAllWindows()
