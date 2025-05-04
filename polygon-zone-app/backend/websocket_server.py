# backend/websocket_server.py

import asyncio
import json
import websockets
from typing import Dict, Any

from processing import process_video_stream
from config import WS_HOST, WS_PORT
from ultralytics import YOLO # Model needs to be accessible here or passed
import cv2

# --- Backend State Management ---
# Hold the currently connected websocket (simple single client assumption)
connected_websocket = None
# Hold the configuration received from the frontend
current_config: Dict[str, Any] | None = None
# Hold the reference to the running processing task
processing_task: asyncio.Task | None = None
# Event to signal the processing task to stop
stop_processing_event = asyncio.Event()

# Global reference to the loaded YOLO model (loaded in main.py)
yolo_model: YOLO | None = None
# Global reference to the video path (from main.py)
video_source_path: str | int | None = None


def set_global_model_and_video_path(model: YOLO, video_path: str | int):
    """Sets the global model and video path to be used by the server/processing."""
    global yolo_model, video_source_path
    yolo_model = model
    video_source_path = video_path
    print(f"Backend initialized with model and video source: {video_source_path}")


# --- WebSocket Handling Logic ---
async def websocket_handler(websocket, path):
    """Handles incoming WebSocket connections and messages."""
    global connected_websocket, current_config, processing_task, stop_processing_event, yolo_model, video_source_path

    if connected_websocket:
        # Reject new connections if one is already active
        print("Rejecting new connection, one client already connected.")
        await websocket.close(code=1008, reason="Another client is already connected")
        return

    connected_websocket = websocket
    print(f"Client connected from {websocket.remote_address}")

    # Send initial status
    await send_status(websocket, "connected")

    try:
        # Listen for messages
        async for message in websocket:
            try:
                data = json.loads(message)
                msg_type = data.get('type')

                print(f"Received message: {msg_type}")

                if msg_type == 'configuration':
                    # Validate and store configuration
                    polygons = data.get('polygons')
                    selected_classes = data.get('selectedClasses')
                    confidence = data.get('confidenceThreshold')

                    if polygons is not None and selected_classes is not None and confidence is not None:
                        current_config = {
                            'polygons': polygons, # Should be list of {'id': str, 'points': list of {'x': float, 'y': float}}
                            'selectedClasses': selected_classes, # Should be list of strings
                            'confidenceThreshold': confidence # Should be float
                        }
                        print("Configuration received and stored.")
                        await send_status(websocket, "config_received")
                    else:
                        print("Invalid configuration message format.")
                        await send_status(websocket, "error", "Invalid configuration format")

                elif msg_type == 'control':
                    action = data.get('action')
                    print(f"Control action: {action}")

                    if action == 'start':
                        if processing_task and not processing_task.done():
                            print("Processing is already running.")
                            await send_status(websocket, "warning", "Processing already active")
                        elif not current_config:
                            print("Start requested but no configuration received.")
                            await send_status(websocket, "error", "Cannot start: No configuration received")
                        elif yolo_model is None or video_source_path is None:
                             print("Backend not fully initialized (model or video source missing).")
                             await send_status(websocket, "error", "Backend not initialized properly")
                        else:
                            print("Starting processing task...")
                            # Reset the stop event for a new run
                            stop_processing_event.clear()
                            # Open video capture here before starting the task
                            cap = cv2.VideoCapture(video_source_path)
                            if not cap.isOpened():
                                print(f"Error: Could not open video source {video_source_path}")
                                await send_status(websocket, "error", f"Could not open video source: {video_source_path}")
                                # Do NOT start task if video cannot be opened
                                cap.release() # Ensure it's released
                                continue # Continue listening for other messages
                            print(f"Video capture opened: {video_source_path}")

                            # Start the processing task
                            processing_task = asyncio.create_task(
                                process_video_stream(
                                    websocket,
                                    cap, # Pass the opened capture object
                                    yolo_model, # Pass the loaded model
                                    current_config,
                                    stop_processing_event
                                )
                            )
                            await send_status(websocket, "started")

                    elif action == 'stop':
                        if processing_task and not processing_task.done():
                            print("Stopping processing task...")
                            stop_processing_event.set() # Signal the task to stop
                            # Wait briefly for the task to acknowledge the stop event
                            await asyncio.sleep(0.1) # Yield control
                            # The process_video_stream function is responsible for sending the final 'stopped' status
                            # and cleaning up (cap.release())
                            # You could also cancel the task directly, but signaling is often cleaner for loops
                            # processing_task.cancel()
                            # await send_status(websocket, "stopping") # Send an intermediate status
                        else:
                            print("No processing task is running.")
                            await send_status(websocket, "warning", "No active processing to stop")
                            # If no task is running, ensure state is clean
                            current_config = None
                            processing_task = None
                            stop_processing_event.clear()
                            await send_status(websocket, "stopped")


                else:
                    print(f"Unknown message type: {msg_type}")
                    await send_status(websocket, "error", f"Unknown message type: {msg_type}")

            except json.JSONDecodeError:
                print("Received invalid JSON.")
                await send_status(websocket, "error", "Invalid JSON received")
            except Exception as e:
                print(f"An unexpected error occurred while handling message: {e}")
                import traceback
                traceback.print_exc()
                await send_status(websocket, "error", f"Internal server error: {str(e)}")

    except websockets.exceptions.ConnectionClosedOK:
        print(f"Client disconnected gracefully from {websocket.remote_address}")
    except websockets.exceptions.ConnectionClosedError as e:
         print(f"Client disconnected with error from {websocket.remote_address}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred in websocket handler: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up on disconnect
        if processing_task and not processing_task.done():
            print("Client disconnected while processing was active. Stopping task.")
            stop_processing_event.set() # Signal the task to stop
            # Do NOT await the task here, it might hang if the client is truly gone.
            # Let it clean itself up in the background.
        current_config = None
        processing_task = None
        stop_processing_event.clear()
        connected_websocket = None # Allow new connections

async def send_status(websocket, status: str, message: str | None = None):
    """Helper function to send a status message to the client."""
    status_payload = {"type": "status", "status": status}
    if message:
        status_payload["message"] = message
    try:
        await websocket.send(json.dumps(status_payload))
    except Exception as e:
        print(f"Failed to send status '{status}': {e}")


async def start_websocket_server(model: YOLO, video_path: str | int):
    """Starts the WebSocket server."""
    set_global_model_and_video_path(model, video_path)
    print(f"Starting WebSocket server on ws://{WS_HOST}:{WS_PORT}")
    async with websockets.serve(websocket_handler, WS_HOST, WS_PORT):
        await asyncio.Future() # Run forever