# backend/processing.py

import asyncio
import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
from typing import List, Dict, Any
import json
import traceback
import time
import os
import websockets # Import websockets for WebSocket handling

from utils import (
    denormalize_polygon_points,
    frontend_classes_to_backend_ids,
    backend_id_to_frontend_string,
    encode_frame_to_base64,
    draw_counters_on_frame,
    cleanup_temp_file
)
# Import necessary config values
from config import OUTPUT_FOLDER, CLASS_NAMES # Ensure OUTPUT_FOLDER is defined in config.py

# Ensure the helper function from server.py is accessible or redefined here
# For simplicity, let's assume it's available via import or defined globally if needed.
# If server.py's send_status is not easily importable, redefine a similar helper here.
async def _send_ws_message(websocket, payload: Dict):
    """Internal helper to send JSON message, handling potential closure."""
    if websocket and not websocket.closed:
        try:
            await websocket.send(json.dumps(payload))
            return True
        except websockets.exceptions.ConnectionClosed:
            print("Warning: WebSocket closed during send.")
            return False
        except Exception as e:
            print(f"Error sending WebSocket message: {e}")
            traceback.print_exc()
            return False
    return False

async def _send_status_update(websocket, status: str, message: str | None = None, extra_data: Dict | None = None):
    """Helper to send status updates via WebSocket."""
    payload = {"type": "status", "status": status}
    if message: payload["message"] = message
    if extra_data: payload.update(extra_data)
    print(f"Sending Status Update: {payload}") # Log the status being sent
    await _send_ws_message(websocket, payload)


# --- Annotation Object Cache ---
class AnnotationCache:
    def __init__(self):
        self.zones: List[sv.PolygonZone] = []
        self.zone_annotators: List[sv.PolygonZoneAnnotator] = []
        self.box_annotators: List[sv.BoxAnnotator] = []
        self.zone_ids: List[str] = []
        try:
            self.colors = sv.ColorPalette.default()
            print("Initialized sv.ColorPalette.default()")
        except AttributeError:
             print("Warning [AnnotationCache]: sv.ColorPalette.default() not found. Creating default palette from hex.")
             default_hex_colors = [ "#FF6464", "#FFB3A4", "#E064FF", "#FF64E0", "#64FF64", "#64E0FF", "#6464FF", "#A4FF64", "#A464FF", "#64A4FF", "#FF64A4", "#E0FF64", "#E064FF" ]
             try: self.colors = sv.ColorPalette.from_hex(default_hex_colors); print(f"Created sv.ColorPalette from {len(default_hex_colors)} hex codes.")
             except Exception as e: print(f"Fatal Error [AnnotationCache]: Failed init sv.ColorPalette: {e}"); traceback.print_exc(); raise RuntimeError("Failed to initialize Supervision ColorPalette.") from e

    def update(self, polygon_configs: List[Dict[str, Any]], frame_resolution_wh: tuple[int, int]):
        self.zones = []; self.zone_annotators = []; self.box_annotators = []; self.zone_ids = []
        print(f"Updating annotation cache for {len(polygon_configs) if polygon_configs else 0} polygon(s) with resolution {frame_resolution_wh}...")
        valid_zones_count = 0
        if not self.colors or not self.colors.colors: print("Error [AnnotationCache]: Color palette empty."); return
        if not polygon_configs: print("No polygon configurations provided."); return
        for index, polygon_config in enumerate(polygon_configs):
            polygon_id = polygon_config.get('id', f'unknown-{index}'); points = polygon_config.get('points')
            if not points or not isinstance(points, list) or len(points) < 3: print(f"Warn [AnnotationCache]: Skipping invalid polygon '{polygon_id}'."); continue
            try:
                pixel_points = denormalize_polygon_points(points, frame_resolution_wh[0], frame_resolution_wh[1])
                if len(pixel_points) < 3: print(f"Warn [AnnotationCache]: Polygon '{polygon_id}' < 3 px points. Skipping."); continue
                if len(np.unique(pixel_points, axis=0)) < 3: print(f"Warn [AnnotationCache]: Polygon '{polygon_id}' degenerate. Skipping."); continue
                zone = sv.PolygonZone(polygon=pixel_points)
                color = self.colors.by_idx(index % len(self.colors.colors))
                self.zone_annotators.append(sv.PolygonZoneAnnotator(zone=zone, color=color, thickness=4, text_thickness=2, text_scale=1))
                self.box_annotators.append(sv.BoxAnnotator(color=color, thickness=3))
                self.zones.append(zone); self.zone_ids.append(polygon_id); valid_zones_count += 1
            except Exception as e: print(f"Error [AnnotationCache]: Failed supervision objects for polygon '{polygon_id}': {e}"); traceback.print_exc()
        print(f"Annotation cache updated with {valid_zones_count} valid zones.")

annotation_cache = AnnotationCache() # Singleton instance

# --- Single Frame Processing Logic ---
async def process_single_frame( frame: np.ndarray, model: YOLO, conf: float, class_filter_backend_ids: List[int], annotation_cache: AnnotationCache ) -> tuple[np.ndarray, Dict[str, Dict[str, int]]]:
    try:
        results_list = model(frame, agnostic_nms=True, verbose=False)
        if not results_list: return frame.copy(), {}
        detections = sv.Detections.from_ultralytics(results_list[0].cpu())
    except Exception as e: print(f"Error [Processing]: YOLO inference: {e}"); traceback.print_exc(); return frame.copy(), {}
    detections = detections[detections.confidence > conf]
    if class_filter_backend_ids and detections.class_id is not None: detections = detections[np.isin(detections.class_id, class_filter_backend_ids)]
    elif class_filter_backend_ids: print("Warn [Processing]: Class IDs requested but detections lack class_id.")
    annotated_frame = frame.copy(); detection_results: Dict[str, Dict[str, int]] = {}
    if not annotation_cache.zones: return annotated_frame, detection_results
    for zone_id, zone, zone_annotator, box_annotator in zip( annotation_cache.zone_ids, annotation_cache.zones, annotation_cache.zone_annotators, annotation_cache.box_annotators ):
        if detections.xyxy is not None and detections.confidence is not None and detections.class_id is not None:
            detections_in_zone = detections[zone.trigger(detections=detections)]
            annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections_in_zone)
            annotated_frame = zone_annotator.annotate(scene=annotated_frame)
            zone_counts: Dict[str, int] = {}
            if len(detections_in_zone.class_id) > 0:
                ids, counts = np.unique(detections_in_zone.class_id, return_counts=True)
                for class_id_backend, count in zip(ids, counts): zone_counts[backend_id_to_frontend_string(int(class_id_backend))] = int(count)
            detection_results[zone_id] = zone_counts
        else: print(f"Warn [Processing]: Skipping zone '{zone_id}' due to missing attrs."); detection_results[zone_id] = {}
    annotated_frame = draw_counters_on_frame(annotated_frame, detection_results); return annotated_frame, detection_results


# --- Main Video Stream Processing Task ---
async def process_video_stream( websocket, video_path: str | int, model: YOLO, config: Dict[str, Any], stop_event: asyncio.Event ):
    print(f"PROCESS_VIDEO_STREAM: Starting for source: {video_path}")
    cap = None; writer = None
    output_filepath = None; output_filename = None
    was_writer_opened_successfully = False
    frame_counter = 0; start_time_loop = time.time()
    final_status = "error"; final_message = "Processing did not start correctly."; extra_final_data = {}

    try:
        # --- Initialization ---
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): raise IOError(f"Could not open video source {video_path}")
        print("Video source opened successfully.")

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        input_fps = float(cap.get(cv2.CAP_PROP_FPS)) if cap.get(cv2.CAP_PROP_FPS) and cap.get(cv2.CAP_PROP_FPS) > 0 else 30.0
        frame_resolution_wh = (frame_width, frame_height)
        if frame_width <= 0 or frame_height <= 0: raise ValueError(f"Invalid video dimensions: {frame_resolution_wh}")
        print(f"Video Res: {frame_resolution_wh}, Input FPS: {input_fps:.2f}")

        # --- Setup Output File Path & Writer ---
        if isinstance(video_path, str):
            try:
                os.makedirs(OUTPUT_FOLDER, exist_ok=True)
                base = os.path.basename(video_path); name, _ = os.path.splitext(base)
                output_filename = f"processed_{name}.mp4"
                output_filepath = os.path.join(OUTPUT_FOLDER, output_filename)
                print(f"Output video target: {output_filepath}")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v'); writer = cv2.VideoWriter(output_filepath, fourcc, input_fps, frame_resolution_wh)
                if writer.isOpened(): print("VideoWriter ready."); was_writer_opened_successfully = True
                else: print("Error: Failed to open VideoWriter."); writer = None # Explicitly set to None
            except Exception as e: print(f"Error setting up output file/writer: {e}"); traceback.print_exc(); writer = None; output_filepath = None; output_filename = None

        # --- Seeking ---
        source_type = config.get('videoSource'); start_frame_time = config.get('startFrameTime', 0.0)
        if source_type == 'file' and isinstance(video_path, str) and start_frame_time > 0:
            start_pos_ms = int(start_frame_time * 1000); print(f"Seeking video to {start_frame_time:.2f}s ({start_pos_ms}ms)")
            if not cap.set(cv2.CAP_PROP_POS_MSEC, start_pos_ms): print("Warn: Seek may not be supported.")
            print(f"Actual position after seek: {cap.get(cv2.CAP_PROP_POS_MSEC):.2f}ms")

        # --- Configuration & Annotation Cache Update ---
        polygon_configs = config.get('polygons', []); selected_classes_frontend = config.get('selectedClasses', [])
        conf_threshold = config.get('confidenceThreshold', 0.4); class_filter_backend_ids = frontend_classes_to_backend_ids(selected_classes_frontend)
        print(f"Config: Conf={conf_threshold}, Polygons={len(polygon_configs)}, Classes={selected_classes_frontend}, BackendIDs={class_filter_backend_ids}")
        annotation_cache.update(polygon_configs, frame_resolution_wh)
        if not annotation_cache.zones: raise ValueError("No valid detection zones configured.")

        # --- Send Started Status ---
        await _send_status_update(websocket, "started")
        print("Sent status: started")

        # --- Processing Loop ---
        print(f"Starting processing loop from time {cap.get(cv2.CAP_PROP_POS_MSEC)/1000:.2f}s...")
        start_time_loop = time.time() # Reset timer just before loop

        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret or frame is None: print(f"Info: End of stream or read error at frame {frame_counter}."); break
            frame_counter += 1

            annotated_frame, detection_results = await process_single_frame( frame, model, conf_threshold, class_filter_backend_ids, annotation_cache )

            if writer and writer.isOpened():
                 try: writer.write(annotated_frame)
                 except Exception as e: print(f"Error writing frame {frame_counter}: {e}") # Log write error

            # Send frame via WS
            encoded_frame = encode_frame_to_base64(annotated_frame)
            if encoded_frame:
                 payload_frame = {"type": "processedFrame", "imageUrl": encoded_frame}
                 if not await _send_ws_message(websocket, payload_frame): stop_event.set(); break # Stop if send fails
                 print(f"DEBUG: Sent frame {frame_counter}") # Keep debug log

            # Send results via WS
            payload_results = {"type": "resultsUpdate", "results": detection_results}
            if not await _send_ws_message(websocket, payload_results): stop_event.set(); break # Stop if send fails
            print(f"DEBUG: Sent results for frame {frame_counter}") # Keep debug log

            await asyncio.sleep(0.0001) # Yield

        # --- Loop Finished ---
        end_time_loop = time.time(); total_duration = end_time_loop - start_time_loop
        processed_fps = frame_counter / total_duration if total_duration > 0 else 0
        print(f"Loop finished. Processed {frame_counter} frames in {total_duration:.2f}s ({processed_fps:.2f} FPS)")

        if stop_event.is_set() and frame_counter > 0:
             final_status = "stopped"; final_message = f"Processing stopped by user after {frame_counter} frames."
        elif frame_counter == 0 and not stop_event.is_set(): # Error before loop or instant end of stream
             final_status = "error"; final_message = "Processing failed to read frames or ended immediately."
        else: # Finished normally
             final_status = "stopped"; final_message = f"Processing finished. Processed {frame_counter} frames."

        # Check if output should have been saved and add filename if it likely was
        if final_status == "stopped" and output_filename and was_writer_opened_successfully:
             # We will add the filename optimistically here. The robust check happens after release.
             extra_final_data['outputFilename'] = output_filename
             final_message += f" Output saving attempted." # Adjusted message

    except (IOError, ValueError) as e: # Catch specific init errors
        print(f"Initialization Error: {e}")
        final_status = "error"; final_message = f"Initialization Error: {str(e)}"
    except asyncio.CancelledError:
        print("Processing task explicitly cancelled.")
        final_status = "stopped"; final_message = "Processing cancelled by user."
    except Exception as e:
        print(f"Unexpected error during processing: {e}")
        traceback.print_exc()
        final_status = "error"; final_message = f"Runtime error: {str(e)}"
    finally:
        # --- Final Cleanup & Status ---
        print("Cleaning up processing resources...")
        writer_released_successfully = False
        if cap and cap.isOpened(): cap.release(); print("Video capture released.")
        if writer and writer.isOpened():
            print("Releasing video writer...")
            writer.release()
            writer_released_successfully = True # Assume release worked if no exception
            print("Video writer released.")
        elif writer: print("Video writer existed but wasn't open.")

        # Robust Check: Verify file exists and has size *after* release
        output_saved_confirmed = False
        if output_filepath and writer_released_successfully:
             # Give a tiny moment for filesystem sync if needed (usually not, but can't hurt)
             await asyncio.sleep(0.1)
             if os.path.exists(output_filepath) and os.path.getsize(output_filepath) > 0:
                 print(f"Confirmed output file saved successfully: {output_filepath}")
                 output_saved_confirmed = True
                 # Ensure filename is in the payload if confirmed saved
                 if output_filename: extra_final_data['outputFilename'] = output_filename
                 # Update message if confirmation differs from optimistic assumption
                 final_message = final_message.replace("Output saving attempted.", "Output saved successfully.")
             else:
                 print(f"Warning: Output file check failed after release for: {output_filepath}")
                 if 'outputFilename' in extra_final_data: del extra_final_data['outputFilename'] # Remove if check failed
                 final_message = final_message.replace("Output saving attempted.", "Error saving output file.")
                 if final_status == "stopped": final_status = "warning" # Downgrade status if save failed? Or keep stopped? Let's keep stopped but adjust message.

        # Cleanup Input File
        if isinstance(video_path, str):
             print(f"Scheduling cleanup for input file: {video_path}")
             asyncio.ensure_future(asyncio.to_thread(cleanup_temp_file, video_path))

        # Send Final Status Update
        print(f"Final Status: {final_status}, Message: {final_message}, Extra: {extra_final_data}")
        await _send_status_update(websocket, final_status, final_message, extra_final_data)
        print("PROCESS_VIDEO_STREAM: Finished.")