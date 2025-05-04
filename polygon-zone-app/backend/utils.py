# backend/utils.py

import numpy as np
import cv2
import base64
import os
import uuid # To generate unique filenames
from typing import List, Dict, Any
import traceback # Import traceback

from config import FRONTEND_CLASS_TO_BACKEND_ID, BACKEND_ID_TO_FRONTEND_CLASS, CLASS_NAMES, UPLOAD_FOLDER, OUTPUT_FOLDER


def cleanup_temp_file(filepath: str):
    """Deletes a temporary file."""
    try:
        if os.path.exists(filepath):
            print(f"Cleaning up temporary file: {filepath}")
            os.remove(filepath)
    except PermissionError:
        print(f"Permission Error: Could not delete file {filepath}. It might still be in use by OpenCV.")
    except OSError as e:
        print(f"OS Error during cleanup of {filepath}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during cleanup of {filepath}: {e}")
        traceback.print_exc()

def denormalize_polygon_points(normalized_points: List[Dict[str, float]], frame_width: int, frame_height: int) -> np.ndarray:
    """
    Converts normalized polygon points ({'x': 0-1, 'y': 0-1}) to pixel coordinates.

    Args:
        normalized_points: List of dictionaries with 'x' and 'y' keys (0-1 range).
        frame_width: The width of the frame in pixels.
        frame_height: The height of the frame in pixels.

    Returns:
        A NumPy array of shape (N, 2) with integer pixel coordinates [[x1, y1], [x2, y2], ...].
    """
    polygon_points = []
    for point in normalized_points:
        # Ensure points are within 0-1 range, clip if necessary and handle missing keys
        x = max(0.0, min(1.0, point.get('x', 0.0))) # Use .get with default for safety
        y = max(0.0, min(1.0, point.get('y', 0.0)))

        pixel_x = int(x * frame_width)
        pixel_y = int(y * frame_height)

        # Clip pixel coordinates to frame boundaries
        pixel_x = max(0, min(frame_width - 1, pixel_x))
        pixel_y = max(0, min(frame_height - 1, pixel_y))

        polygon_points.append([pixel_x, pixel_y])
    # Ensure numpy array has consistent shape, even if points list is empty
    return np.array(polygon_points, dtype=np.int32) if polygon_points else np.empty((0, 2), dtype=np.int32)


def frontend_classes_to_backend_ids(frontend_class_strings: List[str]) -> List[int]:
    """
    Maps a list of frontend class name strings to backend integer class IDs.

    Args:
        frontend_class_strings: List of class names (e.g., ["person", "car"]).

    Returns:
        List of corresponding backend integer IDs (e.g., [0, 2]). Ignores unknown classes.
    """
    backend_ids = []
    for class_name in frontend_class_strings:
        if class_name in FRONTEND_CLASS_TO_BACKEND_ID:
            backend_ids.append(FRONTEND_CLASS_TO_BACKEND_ID[class_name])
        else:
            print(f"Warning [Utils]: Unknown class name received from frontend: '{class_name}'")
    return backend_ids

def backend_id_to_frontend_string(backend_id: int) -> str:
    """
    Maps a backend integer class ID to its frontend class name string.

    Args:
        backend_id: The integer class ID from the backend.

    Returns:
        The corresponding frontend class name string, or the string representation
        of the ID if not found.
    """
    return BACKEND_ID_TO_FRONTEND_CLASS.get(backend_id, str(backend_id))


def encode_frame_to_base64(frame: np.ndarray) -> str:
    """
    Encodes a NumPy array representing an image frame into a base64 string.

    Args:
        frame: The image frame as a NumPy array (H, W, C).

    Returns:
        A base64 encoded string with the data URL prefix, or empty string on failure.
    """
    try:
        # Encode frame as JPEG with quality 70
        # Ensure frame is in a format compatible with imencode (e.g., uint8)
        if frame.dtype != np.uint8:
             frame = frame.astype(np.uint8)

        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        if not ret:
            print("Error [Utils]: Error encoding frame to JPEG.")
            return "" # Return empty string on failure

        # Convert to base64 string and add data URL prefix
        base64_string = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/jpeg;base64,{base64_string}"
    except Exception as e:
        print(f"Exception [Utils]: during frame encoding: {e}")
        traceback.print_exc()
        return ""


def draw_counters_on_frame(frame: np.ndarray, detection_results: Dict[str, Dict[str, int]]) -> np.ndarray:
    """
    Draws detection counts per zone and class onto the frame.

    Args:
        frame: The frame to draw on.
        detection_results: Dictionary { zone_id: { class_id(frontend string): count, ... }, ... }.

    Returns:
        The frame with counters drawn.
    """
    frame_with_counters = frame.copy() # Draw on a copy
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_thickness = 2
    text_color = (0, 0, 0) # Black text in BGR
    background_color = (135, 206, 235) # Light orange/yellow background in BGR (Note: OpenCV uses BGR)
    line_spacing = 25 # Vertical space between text lines
    side_margin = 20
    top_margin = 20

    y_offset = top_margin # Starting Y position from the top

    # Sort zones for consistent display order (sort by number if Zone X format, else alphabetically)
    def sort_key(zone_id):
        try:
            # Attempt to sort by number if zone ID looks like "Zone N"
            if zone_id.startswith('Zone '):
                 num_str = zone_id[len('Zone '):].strip()
                 if num_str.isdigit():
                      return (0, int(num_str)) # Tuple for stable sort, 0 for Zone format
            return (1, zone_id) # Tuple for stable sort, 1 for other formats (alphabetical)
        except ValueError:
            return (1, zone_id) # Fallback to alphabetical sort if parsing fails
        except Exception:
             return (1, zone_id) # Catch any other errors

    sorted_zone_ids = sorted(detection_results.keys(), key=sort_key)


    for zone_id in sorted_zone_ids:
        counts = detection_results.get(zone_id, {}) # Use .get with default empty dict for safety

        # Draw zone header
        header_text = f"{zone_id}:"
        (header_text_w, header_text_h), header_text_base = cv2.getTextSize(header_text, font, font_scale, font_thickness)
        # Align to the right edge with margin
        header_x = frame_with_counters.shape[1] - header_text_w - side_margin
        # header_y is the baseline, adjust for drawing background above it
        header_y = y_offset + header_text_h # Position the text baseline

        # Prepare class lines info and find max width
        class_lines_info = []
        # Sort classes within zone alphabetically by class name string
        sorted_class_ids = sorted(counts.keys())
        for class_id in sorted_class_ids:
            count = counts[class_id]
            count_text = f"- {class_id}: {count}"
            (count_w, count_h), count_base = cv2.getTextSize(count_text, font, font_scale, font_thickness)
            class_lines_info.append((count_text, count_w, count_h, count_base))

        # FIX: Initialize max_line_width before the loop that updates it
        # Initialize with the width of the header text
        max_line_width = header_text_w

        # Now iterate through class lines to find the maximum width
        for info in class_lines_info:
             max_line_width = max(max_line_width, info[1])


        # Calculate background rectangle
        bg_padding = 5 # Consistent padding
        bg_rect_x1 = frame_with_counters.shape[1] - (max_line_width + side_margin + bg_padding * 2)
        # Total height of the text block including spacing and padding
        total_block_height = (header_text_h + header_text_base + bg_padding * 2) # Height for header block
        for info in class_lines_info:
             total_block_height += (info[2] + info[3] + bg_padding * 2) # Height + baseline + padding for each line

        bg_rect_y1 = y_offset - bg_padding # Start background padding above the first line
        bg_rect_y2 = y_offset + total_block_height # End background after last line's padding

        # Ensure coordinates are within frame bounds
        bg_rect_x1 = max(0, bg_rect_x1)
        bg_rect_y1 = max(0, bg_rect_y1)
        bg_rect_x2 = min(frame_with_counters.shape[1], frame_with_counters.shape[1] - side_margin + bg_padding)
        bg_rect_y2 = min(frame_with_counters.shape[0], bg_rect_y2)


        # Draw background for the entire zone block, only if the rectangle is valid
        if bg_rect_x1 < bg_rect_x2 and bg_rect_y1 < bg_rect_y2:
             cv2.rectangle(frame_with_counters, (bg_rect_x1, bg_rect_y1), (bg_rect_x2, bg_rect_y2), background_color, -1)
        else:
             # This warning is less critical, as it just means the counter couldn't be drawn neatly
             print(f"Warning [Utils]: Skipping drawing background for zone '{zone_id}' due to invalid rectangle coordinates or dimensions: ({bg_rect_x1}, {bg_rect_y1}) to ({bg_rect_x2}, {bg_rect_y2}) on frame {frame_with_counters.shape[:2]}.")


        # Draw zone header text
        cv2.putText(frame_with_counters, header_text, (header_x, header_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

        # Move down for class counts text drawing
        current_y = y_offset + (header_text_h + header_text_base + bg_padding * 2) # Position after header block

        # Draw class count texts
        for count_text, count_w, count_h, count_base in class_lines_info:
            count_x = frame_with_counters.shape[1] - count_w - side_margin
            count_y = current_y + count_h # Position the text baseline

            cv2.putText(frame_with_counters, count_text, (count_x, count_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
            current_y += (count_h + count_base + bg_padding * 2) # Move down for next class count


        y_offset = bg_rect_y2 + 10 # Set next zone's starting Y below this block + margin


    return frame_with_counters


def generate_upload_filepath(original_filename: str) -> str:
    """Generates a unique filepath in the UPLOAD_FOLDER for an uploaded file."""
    _, file_extension = os.path.splitext(original_filename)
    # Use uuid to ensure unique name, keep original extension, ensure extension exists
    ext = file_extension.lower() if file_extension else ".tmp" # Use .tmp if no extension, use lowercase
    temp_filename = f"{uuid.uuid4()}{ext}"
    # FIX: Use UPLOAD_FOLDER from config
    return os.path.join(UPLOAD_FOLDER, temp_filename)

# Note: No change needed for cleanup_temp_file, it takes a path and deletes it


# Note: Function to generate an output filepath could be added here later if needed
# def generate_output_filepath(original_filename: str) -> str:
#     """Generates a unique filepath in the OUTPUT_FOLDER for a processed output file."""
#     base_name, file_extension = os.path.splitext(original_filename)
#     ext = file_extension.lower() if file_extension else ".mp4" # Default to mp4 for output
#     output_filename = f"{base_name}_annotated_{uuid.uuid4()}{ext}"
#     # FIX: Use OUTPUT_FOLDER from config
#     return os.path.join(OUTPUT_FOLDER, output_filename)