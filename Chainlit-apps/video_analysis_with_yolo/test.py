import chainlit as cl
import cv2
import tempfile
import os
from typing import List, Dict
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt
import ollama


# Load YOLOv8 model
yolo_model = YOLO("yolo11n.pt")  # Replace with your YOLOv8 model path if needed


def process_video_with_visualization(video_path: str, output_path: str) -> List[Dict[str, any]]:
    """
    Process the video with YOLOv8 and generate a processed output video.
    """
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Define video writer for processed output
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    detections = []

    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        # YOLOv8 detection
        results = yolo_model(frame)
        frame_detections = []
        for result in results:  # Process each detection
            for detection in result.boxes.data:
                obj_name = yolo_model.names[int(detection[5])]
                confidence = float(detection[4])
                bbox = detection[:4].tolist()
                frame_detections.append({
                    "name": obj_name,
                    "confidence": confidence,
                    "bbox": bbox,
                })

                # Draw bounding box and label on the frame
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
                cv2.putText(frame, f"{obj_name} ({confidence:.2f})", (int(bbox[0]), int(bbox[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        detections.append({
            "frame_number": i,
            "timestamp": i / fps,
            "objects": frame_detections
        })

        # Write the processed frame to the output video
        out.write(frame)

    cap.release()
    out.release()
    return detections


def generate_density_map(detections: List[Dict], frame_shape: tuple, output_path: str):
    """
    Generate a density map based on object detections.
    """
    heatmap = np.zeros(frame_shape[:2], dtype=np.float32)
    for result in detections:
        for obj in result["objects"]:
            bbox = obj["bbox"]
            cx, cy = int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2)
            if 0 <= cx < frame_shape[1] and 0 <= cy < frame_shape[0]:
                heatmap[cy, cx] += 1

    # Normalize and create a density map
    heatmap = cv2.GaussianBlur(heatmap, (15, 15), 0)
    plt.imshow(heatmap, cmap="hot", interpolation="nearest")
    plt.title("Density Map")
    plt.colorbar()
    plt.savefig(output_path)
    plt.close()


async def llm_summarize(detections: List[Dict], video_context: str = "traffic analysis") -> str:
    """
    Summarize YOLOv8 results using an LLM.
    """
    prompt = (
        f"You are an expert in {video_context}. Based on the following object detection results from YOLOv8, "
        "analyze the traffic video. Summarize key events, patterns, and any anomalies step by step:\n\n"
    )
    for result in detections:
        objects = ", ".join([f"{obj['name']} (Confidence: {obj['confidence']:.2f})" for obj in result["objects"]])
        prompt += f"- Frame {result['frame_number']} @ {result['timestamp']:.2f}s: Detected objects: {objects}\n"

    prompt += "\nProvide a detailed and coherent narrative about the traffic activity in the video."

    response = ollama.chat(
        model="llama3.2-vision",  # Replace with the correct model name
        messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]


@cl.on_chat_start
async def start():
    """Welcome message"""
    await cl.Message(
        content="Welcome! Please upload a traffic video, and I'll analyze it frame by frame."
    ).send()


@cl.on_message
async def main(message: cl.Message):
    """Main handler for video analysis."""
    video_elements = [elem for elem in message.elements if elem.mime.startswith("video/")]
    if not video_elements:
        await cl.Message(content="Please upload a traffic video to analyze.").send()
        return

    video = video_elements[0]
    video_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    processed_video_path = "output.mp4"  # Custom name for output video
    density_map_path = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name

    with open(video_path, "wb") as file:
        file.write(video.content if video.content else open(video.path, "rb").read())

    # Process video with visualization
    await cl.Message(content="Processing the video with YOLOv8...").send()
    detections = process_video_with_visualization(video_path, processed_video_path)

    # Generate density map
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if ret:
        generate_density_map(detections, frame.shape, density_map_path)
    cap.release()

    # Summarize results using LLM
    await cl.Message(content="Summarizing the video using AI...").send()
    summary = await llm_summarize(detections)

    # Display processed video and density map
    await cl.Message(
        content="Here is the processed video with object detections:",
        elements=[
            cl.Video(name="Processed Video", path=processed_video_path, display="inline"),
        ]
    ).send()

    await cl.Message(
        content="Here is the density map showing object distribution:",
        elements=[
            cl.Image(name="Density Map", path=density_map_path, display="inline"),
        ]
    ).send()

    # Display the summary
    await cl.Message(content=f"Video Summary:\n\n{summary}").send()

    # Cleanup temporary files
    os.unlink(video_path)
    os.unlink(density_map_path)

    # Uncomment if speed estimation is needed in the future
    # process_video_with_speed_estimation(video_path, speed_estimation_path, fps=int(cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FPS)))
