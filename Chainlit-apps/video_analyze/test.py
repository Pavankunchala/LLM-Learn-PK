import chainlit as cl
import ollama
import cv2
import tempfile
import os
from typing import List
import numpy as np
from PIL import Image
import io

@cl.on_chat_start
async def start():
    await cl.Message(
        content="Welcome! Please send a video and I'll analyze it frame by frame."
    ).send()

def extract_frames(video_data: bytes, max_frames: int = 5) -> List[bytes]:
    """Extract frames from video bytes data"""
    # Create temporary file to store video
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_video:
        temp_video.write(video_data)
        temp_video_path = temp_video.name

    frames = []
    try:
        # Open video file
        cap = cv2.VideoCapture(temp_video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame interval to get evenly spaced frames
        frame_interval = max(total_frames // max_frames, 1)
        
        frame_count = 0
        while len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_interval == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Convert to PIL Image
                pil_image = Image.fromarray(frame_rgb)
                # Convert to bytes
                img_byte_arr = io.BytesIO()
                pil_image.save(img_byte_arr, format='JPEG')
                frames.append(img_byte_arr.getvalue())
                
            frame_count += 1
            
        cap.release()
    finally:
        # Clean up temporary file
        os.unlink(temp_video_path)
        
    return frames

@cl.on_message
async def main(message: cl.Message):
    # Get video elements from the message
    video_elements = [elem for elem in message.elements if elem.mime.startswith('video/')]

    if not video_elements:
        await cl.Message(
            content="Please provide a video to analyze."
        ).send()
        return

    # Process each video element
    for video in video_elements:
        try:
            # Get video content
            if video.path:
                with open(video.path, 'rb') as file:
                    video_data = file.read()
            else:
                video_data = video.content

            # Extract frames from video
            frames = extract_frames(video_data)

            # Create a markdown message to show progress
            progress_msg = cl.Message(content="Analyzing video frames...")
            await progress_msg.send()

            # Analyze each frame with history
            frame_analyses = []
            previous_descriptions = ""
            for i, frame in enumerate(frames):
                # Prompt for individual frame analysis with history
                frame_prompt = (
                    f"You are an expert in video analysis. Carefully analyze frame {i+1}. "
                    f"Here is the history of the previous frames:\n{previous_descriptions}\n"
                    f"Describe the objects, actions, and context in this frame. "
                    f"Relate it to the previous frames where relevant. "
                    f"Think step by step and provide a detailed explanation."
                )

                response = ollama.chat(
                    model='llama3.2-vision',
                    messages=[
                        {
                            'role': 'user',
                            'content': frame_prompt,
                            'images': [frame],
                        },
                    ],
                )

                frame_analysis = response['message']['content']
                frame_analyses.append(f"Frame {i+1}: {frame_analysis}")
                previous_descriptions += f"Frame {i+1}: {frame_analysis}\n"

                # Show the frame and its analysis
                await cl.Message(
                    content=f"Frame {i+1} Analysis:\n{frame_analysis}",
                    elements=[
                        cl.Image(
                            name=f"Frame {i+1}",
                            content=frame,
                            display="inline"
                        )
                    ]
                ).send()

            # Generate a consolidated analysis for the entire video
            summary_prompt = (
                "Based on the analysis of all frames provided below, summarize the video:\n"
                f"{previous_descriptions}\n"
                "Provide a detailed summary of the key events, transitions, and overall context. "
                "Think step by step and ensure the narrative is clear."
            )

            consolidated_response = ollama.chat(
                model='llama3.2-vision',
                messages=[
                    {
                        'role': 'user',
                        'content': summary_prompt,
                    },
                ],
            )

            video_summary = consolidated_response['message']['content']

            # Send the final summary
            await cl.Message(
                content=f"Video Analysis Summary:\n\n{video_summary}"
            ).send()

        except Exception as e:
            await cl.Message(
                content=f"Error processing video: {str(e)}"
            ).send()
