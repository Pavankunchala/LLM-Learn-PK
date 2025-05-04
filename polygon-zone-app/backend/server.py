# backend/server.py

import asyncio
import json
import websockets

import os # For path joining
from typing import Dict, Any
import uuid # For unique file names
from aiohttp import web
import aiohttp_cors
import traceback # Import traceback

# FIX: Import generate_upload_filepath instead of generate_temp_filepath from utils
# FIX: Import UPLOAD_FOLDER, OUTPUT_FOLDER from config
from processing import process_video_stream # Removed AnnotationCache import as it's not used here
from config import WS_HOST, WS_PORT, HTTP_HOST, HTTP_PORT, UPLOAD_FOLDER, MAX_FILE_SIZE_MB, ALLOWED_ORIGINS, OUTPUT_FOLDER
from utils import generate_upload_filepath, cleanup_temp_file # Use utility for path/cleanup
from ultralytics import YOLO

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


def set_global_model(model: YOLO):
    """Sets the global model to be used by processing."""
    global yolo_model
    yolo_model = model
    print("Backend initialized with model.")


# --- HTTP Upload Handling ---
async def upload_handler(request: web.Request):
    """Handles video file uploads via HTTP POST."""
    print(f"Received upload request from {request.remote}")

    # Check content type
    if request.content_type != 'multipart/form-data':
         print("Upload rejected: Unsupported Media Type.")
         return web.Response(status=415, text="Unsupported Media Type: Must be multipart/form-data")

    reader = await request.multipart()

    # Read file part
    file_part = await reader.next()
    if file_part is None or file_part.name != 'videoFile': # 'videoFile' should match frontend form field name
         print("Upload rejected: Missing 'videoFile' part.")
         return web.Response(status=400, text="Missing 'videoFile' part in multipart request")

    filename = file_part.filename
    if not filename:
        print("Upload rejected: Missing filename.")
        return web.Response(status=400, text="Missing filename")

    # FIX: Use generate_upload_filepath from utils
    temp_filepath = generate_upload_filepath(filename)

    print(f"Saving uploaded file temporarily to: {temp_filepath}")

    # Save the file in chunks to prevent excessive memory usage
    size = 0
    try:
        # Ensure the directory for the file exists (in case UPLOAD_FOLDER wasn't created for some reason)
        os.makedirs(os.path.dirname(temp_filepath), exist_ok=True)
        with open(temp_filepath, 'wb') as f:
            while True:
                chunk = await file_part.read_chunk()
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_FILE_SIZE_MB * 1024 * 1024:
                    print(f"Upload exceeded max size {MAX_FILE_SIZE_MB}MB. Size: {size} bytes")
                    # Clean up the partial file
                    cleanup_temp_file(temp_filepath) # Use cleanup_temp_file from utils
                    return web.Response(status=413, text=f"File size exceeds {MAX_FILE_SIZE_MB}MB limit")
                f.write(chunk)

        print(f"Successfully saved file: {filename} ({size} bytes) to {temp_filepath}")
        # Return the temporary path to the frontend
        return web.json_response({'status': 'success', 'filePath': temp_filepath, 'filename': filename})

    except Exception as e:
        print(f"Error during file upload: {e}")
        traceback.print_exc()
        # Clean up in case of error
        cleanup_temp_file(temp_filepath) # Use cleanup_temp_file from utils
        return web.Response(status=500, text=f"Failed to save file: {e}")


# --- WebSocket Handling Logic ---
async def websocket_handler(websocket, path):
    """Handles incoming WebSocket connections and messages."""
    global connected_websocket, current_config, processing_task, stop_processing_event, yolo_model

    if connected_websocket:
        # Reject new connections if one is already active
        print(f"Rejecting new connection attempt from {websocket.remote_address}, one client already connected.")
        try:
             # Use 1000 for normal closure as recommended
             await websocket.close(code=1000, reason="Another client is already connected")
        except Exception: pass # Ignore errors on closing connection
        return

    connected_websocket = websocket
    print(f"Client connected from {websocket.remote_address}")

    # Send initial status
    try:
        await send_status(websocket, "connected", message="WebSocket connected")
        # Optionally send available classes here if not hardcoded in frontend
        # await websocket.send(json.dumps({"type": "availableClasses", "classes": config.CLASS_NAMES}))
    except Exception as e:
        print(f"Failed to send initial 'connected' status: {e}")
        connected_websocket = None # Reset state if initial send fails
        try: await websocket.close() # Attempt to close the connection
        except Exception: pass
        return


    try:
        # Listen for messages
        async for message in websocket:
            try:
                data = json.loads(message)
                msg_type = data.get('type')

                # print(f"Received message type: {msg_type}") # Too verbose, log specific actions

                if msg_type == 'configuration':
                    print("Received configuration message.")
                    # Validate and store configuration
                    polygons = data.get('polygons') # List of {'id': str, 'points': list of {'x': float, 'y': float}}
                    selected_classes = data.get('selectedClasses') # List of strings
                    confidence = data.get('confidenceThreshold') # Float
                    video_source = data.get('videoSource') # 'webcam' or 'file'
                    video_path_or_index = data.get('videoPath') # String path or int index (0 for webcam)
                    start_frame_time = data.get('startFrameTime', 0.0) # Get start time, default to 0

                    print(f"  - Source: {video_source}")
                    print(f"  - Path/Index: {video_path_or_index}")
                    print(f"  - Polygons: {len(polygons) if polygons is not None else 0}")
                    print(f"  - Classes: {selected_classes}") # Log actual classes
                    print(f"  - Confidence: {confidence}")
                    print(f"  - Start Frame Time: {start_frame_time}")


                    if (polygons is not None and isinstance(polygons, list) and
                        selected_classes is not None and isinstance(selected_classes, list) and
                        confidence is not None and isinstance(confidence, (int, float)) and
                        video_source in ['webcam', 'file'] and
                        video_path_or_index is not None and
                        isinstance(start_frame_time, (int, float))): # Basic validation, including start_frame_time type

                        current_config = {
                            'polygons': polygons,
                            'selectedClasses': selected_classes,
                            'confidenceThreshold': float(confidence), # Ensure float
                            'videoSource': video_source,
                            'videoPath': video_path_or_index, # This will be the upload path or 0
                            'startFrameTime': float(start_frame_time) # Ensure float
                        }
                        print("Configuration received and stored successfully.")
                        await send_status(websocket, "config_received", message="Configuration received")
                    else:
                        print("Invalid configuration message format received.")
                        await send_status(websocket, "error", "Invalid configuration format")

                elif msg_type == 'control':
                    action = data.get('action')
                    print(f"Received control action: {action}")

                    if action == 'start':
                        if processing_task and not processing_task.done():
                            print("Start requested, but processing is already running.")
                            await send_status(websocket, "warning", "Processing already active")
                        elif not current_config:
                            print("Start requested but no configuration received.")
                            await send_status(websocket, "error", "Cannot start: No configuration received")
                        elif yolo_model is None:
                             print("Start requested, but backend not initialized (model missing).")
                             await send_status(websocket, "error", "Backend not initialized properly")
                        # Check if video source path is available in config
                        elif current_config.get('videoPath') is None:
                             print("Start requested, but videoPath is missing in the current configuration.")
                             await send_status(websocket, "error", "Video source path missing in configuration.")
                        else:
                            print("Initiating processing task start...")
                            # Reset the stop event for a new run
                            stop_processing_event.clear()

                            # Get video source from current config
                            source_path = current_config.get('videoPath')

                            # Start the processing task
                            processing_task = asyncio.create_task(
                                process_video_stream(
                                    websocket,
                                    source_path, # Pass the video path/index
                                    yolo_model, # Pass the loaded model
                                    current_config, # Pass the full config
                                    stop_processing_event
                                )
                            )
                            # The 'started' status is sent from process_video_stream after it successfully opens the video
                            # This prevents sending 'started' if video opening fails.

                    elif action == 'stop':
                        print("Received stop control action.")
                        if processing_task and not processing_task.done():
                            print("Stopping processing task...")
                            stop_processing_event.set() # Signal the task to stop
                            # Wait briefly for the task to acknowledge and clean up
                            try:
                                # Wait for the processing task to finish after receiving stop signal
                                # Ensure processing_task is awaited to allow its 'finally' block to run (including cleanup)
                                await asyncio.wait_for(processing_task, timeout=10.0) # Wait max 10s
                                print("Processing task stopped gracefully.")
                            except asyncio.TimeoutError:
                                print("Processing task did not stop in time, cancelling.")
                                processing_task.cancel() # Force cancel if it doesn't stop gracefully
                                print("Processing task cancelled.")
                            except asyncio.CancelledError:
                                print("Processing task was already cancelled.")
                            except Exception as e:
                                print(f"An error occurred while waiting for processing task to stop: {e}")
                                traceback.print_exc()

                            # Clean up task reference regardless of how it stopped
                            processing_task = None
                            # The 'stopped' status is sent from process_video_stream in its finally block

                        else:
                            print("Stop requested, but no processing task is running.")
                            await send_status(websocket, "warning", "No active processing to stop")
                            # If no task is running, ensure state is clean
                            # Cleanup the associated uploaded file if it exists and wasn't processed
                            if current_config and current_config.get('videoSource') == 'file':
                                 video_path = current_config.get('videoPath')
                                 # Use cleanup_temp_file from utils
                                 if isinstance(video_path, str) and os.path.exists(video_path): # Check if it's a path string and exists
                                     print("Cleaning up dormant uploaded file.")
                                     asyncio.ensure_future(asyncio.to_thread(cleanup_temp_file, video_path)) # Cleanup without blocking

                            current_config = None
                            processing_task = None
                            stop_processing_event.clear()
                            # Send stopped status explicitly as processing task didn't run
                            await send_status(websocket, "stopped")


                else:
                    print(f"Unknown message type: {msg_type}")
                    await send_status(websocket, "error", f"Unknown message type: {msg_type}")

            except json.JSONDecodeError:
                print("Received invalid JSON message.")
                await send_status(websocket, "error", "Invalid JSON received")
            except Exception as e:
                print(f"An unexpected error occurred while handling message: {e}")
                traceback.print_exc()
                await send_status(websocket, "error", f"Internal server error: {str(e)}")

    except websockets.exceptions.ConnectionClosedOK:
        print(f"Client disconnected gracefully from {websocket.remote_address}")
    except websockets.exceptions.ConnectionClosedError as e:
         print(f"Client disconnected with error from {websocket.remote_address}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred in websocket handler: {e}")
        traceback.print_exc()
    finally:
        print(f"Cleaning up state for disconnected client {websocket.remote_address}")
        # Clean up on disconnect
        if processing_task and not processing_task.done():
            print("Client disconnected while processing was active. Signalling task to stop.")
            stop_processing_event.set() # Signal the task to stop
            # Do NOT await the task here, it will run its finally block independently.

        # Cleanup the associated uploaded file if client disconnects before processing finishes
        if current_config and current_config.get('videoSource') == 'file':
             video_path = current_config.get('videoPath')
             # Use cleanup_temp_file from utils
             if isinstance(video_path, str) and os.path.exists(video_path): # Check if it's a path string and exists
                 print("Cleaning up uploaded file on client disconnect.")
                 # Use ensure_future to schedule cleanup without blocking the handler's exit
                 asyncio.ensure_future(asyncio.to_thread(cleanup_temp_file, video_path))


        # Reset shared state for the next potential connection
        current_config = None
        processing_task = None
        stop_processing_event.clear() # Ensure event is cleared for the next run
        connected_websocket = None # Allow new connections


async def send_status(websocket, status: str, message: str | None = None):
    """Helper function to send a status message to the client."""
    status_payload = {"type": "status", "status": status}
    if message:
        status_payload["message"] = message
    try:
        # Ensure the websocket is still open before sending
        if websocket and not websocket.closed:
             # print(f"Sending status: {status}") # Avoid spamming logs with 'connected' etc.
             await websocket.send(json.dumps(status_payload))
        else:
             print(f"Attempted to send status '{status}' but WebSocket was closed.")
    except Exception as e:
        print(f"Failed to send status '{status}': {e}")
        traceback.print_exc()


async def start_servers(model: YOLO):
    """Starts both the HTTP upload server and the WebSocket server."""
    set_global_model(model)

    # --- Setup HTTP Upload Server ---
    app = web.Application()
    # Configure CORS
    cors = aiohttp_cors.setup(app, defaults={
        # Use ALLOWED_ORIGINS from config
        "*": aiohttp_cors.ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*",
            allow_methods=["POST"], # Only POST for uploads
            max_age=3600, # Cache CORS preflight requests
            # FIX: Changed allow_origin back to allow_origins to match the error in your environment
            # allow_origins=ALLOWED_ORIGINS # Use allowed origins
        )
    })
    # Add upload route
    resource = cors.add(app.router.add_resource("/upload"))
    cors.add(resource.add_route("POST", upload_handler))

    runner = web.AppRunner(app)
    await runner.setup()
    http_site = web.TCPSite(runner, HTTP_HOST, HTTP_PORT)
    print(f"Starting HTTP upload server on http://{HTTP_HOST}:{HTTP_PORT}")
    await http_site.start()

    # --- Setup WebSocket Server ---
    print(f"Starting WebSocket server on ws://{WS_HOST}:{WS_PORT}")
    ws_server = await websockets.serve(websocket_handler, WS_HOST, WS_PORT)

    # Keep the servers running indefinitely
    print("Servers are running. Press Ctrl+C to stop.")
    await asyncio.Future() # Run forever