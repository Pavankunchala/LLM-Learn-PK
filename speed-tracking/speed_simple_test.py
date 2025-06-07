import cv2
import numpy as np
import math
import time
import argparse
import logging
from collections import deque
from ultralytics import YOLO
import supervision as sv

# --- Logging Setup ---
# Configure basic logging for errors and info
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# --- Constants ---
SPEED_THRESHOLD_KMH = 25.0  # Speed threshold to trigger red box
DEFAULT_COLOR = (0, 255, 0)  # Green (BGR) for vehicles below threshold
SPEEDING_COLOR = (0, 0, 255) # Red (BGR) for vehicles above threshold
TEXT_COLOR = (255, 255, 255) # White for text
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
LINE_THICKNESS = 2
MAX_TRAJECTORY_LENGTH = 5 # Use a slightly longer history for better raw speed input stability (5 points)
SPEED_SMOOTHING_FACTOR = 0.6 # Factor for Exponential Moving Average (0.0 = no smoothing, 1.0 = no change). Adjusted for potentially smoother results.
MIN_DISTANCE_CHANGE_PX_FOR_SPEED = 2.0 # Ignore movement less than this many pixels over history for meaningful speed
MIN_TIME_CHANGE_S_FOR_SPEED = 0.05 # Minimum time difference over history for speed calculation (>1 frame at 30fps)


# Define the classes we care about (vehicles)
TARGET_CLASSES = ['car', 'truck', 'bus', 'motorcycle', 'bicycle']

# --- Simple Speed Tracker Class ---
class SimpleSpeedTracker:
    def __init__(self, model_path, conf_threshold, pixel_to_meter, frame_rate):
        """
        Initializes the SimpleSpeedTracker.

        Args:
            model_path (str): Path to the YOLO model file.
            conf_threshold (float): Confidence threshold for object detection.
            pixel_to_meter (float): Conversion factor from pixels to meters.
            frame_rate (float): Frame rate of the video source.
        """
        self.model = self._load_model(model_path)
        self.tracker = self._setup_tracker(frame_rate)
        self.conf_threshold = conf_threshold
        self.pixel_to_meter = pixel_to_meter
        self.frame_rate = frame_rate

        # Data structures to store track information
        # track_history stores deque of (midpoint_px, timestamp_s) for raw speed input
        self.track_history = {} # Stores deque of (midpoint_px, timestamp_s) for each track ID
        # track_data stores smoothed speed and other relevant info
        self.track_data = {}    # Stores {'speed_kmh': float, 'velocity_mps': np.ndarray}

    def _load_model(self, model_path):
        """Loads the YOLO model."""
        logger.info(f"Loading YOLO model: {model_path}")
        try:
            model = YOLO(model_path)
            logger.debug("YOLO model loaded.")
            return model
        except Exception as e:
            logger.error(f"Failed to load YOLO model '{model_path}': {e}", exc_info=True)
            raise # Re-raise the exception as we cannot proceed without a model

    def _setup_tracker(self, frame_rate):
        """Sets up the ByteTrack tracker."""
        logger.info(f"Setting up ByteTrack with frame rate: {frame_rate}")
        # Use ByteTrack defaults, providing the frame rate
        # Adjust tracker parameters slightly for potentially better stability if needed
        return sv.ByteTrack(
            frame_rate=int(frame_rate),
            track_activation_threshold=0.3, # Slightly higher threshold than default 0.25
            lost_track_buffer=int(frame_rate * 1.5), # Keep lost tracks slightly longer
            minimum_matching_threshold=0.9 # Higher threshold for matching existing tracks
        )

    def _midpoint(self, box):
        """Calculates the midpoint of a bounding box."""
        x1, y1, x2, y2 = map(int, box[:4])
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def _calculate_speed_and_velocity(self, track_hist_entry, p2m_ratio, previous_smoothed_data):
        """
        Calculates raw speed/velocity from track history (using first and last point in deque),
        then applies smoothing.

        Args:
            track_hist_entry (deque): Deque of (point_px, timestamp_s) tuples (maxlen MAX_TRAJECTORY_LENGTH).
            p2m_ratio (float): Pixel to meter conversion ratio.
            previous_smoothed_data (dict): Dictionary containing 'speed_kmh' and 'velocity_mps'
                                           from the previous frame for smoothing.

        Returns:
            tuple: (smoothed speed in km/h, smoothed velocity vector in m/s).
        """
        # Need at least two points to calculate speed
        if len(track_hist_entry) < 2:
            # If less than 2 points, return previous smoothed values or zero
            return previous_smoothed_data.get('speed_kmh', 0.0), previous_smoothed_data.get('velocity_mps', np.array([0.0, 0.0]))

        # Use the oldest and newest points in the history deque for calculation
        (p1px, t1) = track_hist_entry[0] # Oldest point
        (p2px, t2) = track_hist_entry[-1] # Newest point

        dt_s = t2 - t1
        # Avoid division by zero or near-zero time difference
        if dt_s < MIN_TIME_CHANGE_S_FOR_SPEED:
            # If time difference is too small, speed is unstable or object is stopped.
            # Return previous smoothed values.
            return previous_smoothed_data.get('speed_kmh', 0.0), previous_smoothed_data.get('velocity_mps', np.array([0.0, 0.0]))

        # Calculate displacement in pixels
        dx_px = p2px[0] - p1px[0]
        dy_px = p2px[1] - p1px[1]

        # Calculate distance in pixels
        dist_px = math.hypot(dx_px, dy_px)

        # Ignore negligible movement over the history duration to reduce noise for nearly stationary objects
        if dist_px < MIN_DISTANCE_CHANGE_PX_FOR_SPEED:
             raw_speed_kmh = 0.0
             raw_velocity_mps = np.array([0.0, 0.0])
        else:
            # Convert distance to meters
            dist_m = dist_px * p2m_ratio

            # Calculate raw speed in meters per second and convert to km/h
            speed_mps = dist_m / dt_s
            raw_speed_kmh = speed_mps * 3.6

            # Calculate raw velocity vector
            raw_velocity_mps = np.array([dx_px * p2m_ratio / dt_s, dy_px * p2m_ratio / dt_s]) # Fixed typo here

        # Apply Exponential Moving Average (EMA) smoothing
        prev_speed_kmh = previous_smoothed_data.get('speed_kmh', raw_speed_kmh) # Use raw speed as starting point if no previous data
        prev_velocity_mps = previous_smoothed_data.get('velocity_mps', raw_velocity_mps)

        smoothed_speed_kmh = (SPEED_SMOOTHING_FACTOR * prev_speed_kmh + (1 - SPEED_SMOOTHING_FACTOR) * raw_speed_kmh)
        smoothed_velocity_mps = (SPEED_SMOOTHING_FACTOR * prev_velocity_mps + (1 - SPEED_SMOOTHING_FACTOR) * raw_velocity_mps)

        # Ensure speed is non-negative
        smoothed_speed_kmh = max(0.0, smoothed_speed_kmh)

        return smoothed_speed_kmh, smoothed_velocity_mps


    def _cleanup_tracks(self, current_frame_track_ids):
        """Removes track history and data for IDs no longer present in the current frame."""
        tracks_to_del = [tid for tid in self.track_history if tid not in current_frame_track_ids]
        for tid in tracks_to_del:
            if tid in self.track_history:
                del self.track_history[tid]
            if tid in self.track_data:
                del self.track_data[tid]
        # logger.debug(f"Cleaned up {len(tracks_to_del)} old tracks.") # Uncomment for debugging cleanup

    def process_frame(self, frame):
        """
        Processes a single frame: detects, tracks, calculates speed, and annotates.

        Args:
            frame (np.ndarray): The input frame (BGR image).

        Returns:
            np.ndarray: The annotated frame.
        """
        current_time_s = time.time()
        frame_height, frame_width = frame.shape[:2]
        annotated_frame = frame.copy() # Work on a copy

        # 1. Detection
        # Run YOLO model inference. verbose=False to keep console clean.
        # Use imgsz=640 as a common default; can be adjusted if needed
        results = self.model(frame, conf=self.conf_threshold, verbose=False, imgsz=640)[0]
        detections = sv.Detections.from_ultralytics(results)

        # 2. Filter detections by target classes (vehicles)
        # detections.class_id contains integer IDs. model.names maps IDs to strings.
        if detections.class_id is not None:
            filtered_indices = [
                i for i, class_id in enumerate(detections.class_id)
                # Ensure class_id is not None and is in our target list
                if class_id is not None and int(class_id) < len(self.model.names) and self.model.names[int(class_id)] in TARGET_CLASSES
            ]
            detections = detections[filtered_indices]
        else:
             # If no detections at all, create an empty detections object to prevent errors
             detections = sv.Detections.empty()

        # 3. Update tracker with current detections
        # The tracker assigns unique IDs to objects across frames
        # Pass the filtered detections to the tracker
        if len(detections.xyxy) > 0:
             detections = self.tracker.update_with_detections(detections)
        else:
             # If no detections after filtering, ensure tracker still gets an empty update
             detections = self.tracker.update_with_detections(sv.Detections.empty())


        current_frame_track_ids = set()

        # Check if tracker_id is available and matches the number of detections
        # Only proceed if we have tracked objects with IDs
        if detections.tracker_id is not None and len(detections.tracker_id) == len(detections.xyxy):
            # Loop through each tracked object
            for i in range(len(detections.xyxy)):
                track_id = int(detections.tracker_id[i])
                current_frame_track_ids.add(track_id) # Keep track of IDs present in this frame

                box = detections.xyxy[i]
                # class_id_val = detections.class_id[i] # Class name not strictly needed

                mid_px = self._midpoint(box)

                # Ensure track history and data exist for this ID, initialize if new
                if track_id not in self.track_history:
                    self.track_history[track_id] = deque(maxlen=MAX_TRAJECTORY_LENGTH)
                    # Initialize speed and velocity to zero for new tracks
                    self.track_data[track_id] = {'speed_kmh': 0.0, 'velocity_mps': np.array([0.0, 0.0])}

                # Append the current position and timestamp to history
                self.track_history[track_id].append((mid_px, current_time_s))

                # Calculate and smooth speed using the updated history and previous data
                # We pass the track_data entry *before* updating it for this frame
                previous_smoothed_data = self.track_data[track_id]
                smoothed_speed_kmh, smoothed_velocity_mps = self._calculate_speed_and_velocity(
                    self.track_history[track_id],
                    self.pixel_to_meter,
                    previous_smoothed_data
                )

                # Update track_data with the newly smoothed values
                self.track_data[track_id].update({
                    'speed_kmh': smoothed_speed_kmh,
                    'velocity_mps': smoothed_velocity_mps,
                    # 'last_update_time': current_time_s, # Not strictly needed for this simple version
                    # 'last_seen_pos': mid_px # Not strictly needed
                })

                # Get the speed to display and check against the threshold
                # Use the updated smoothed speed
                display_speed_kmh = self.track_data[track_id]['speed_kmh']

                # Determine the color of the bounding box based on speed
                box_color = SPEEDING_COLOR if display_speed_kmh > SPEED_THRESHOLD_KMH else DEFAULT_COLOR

                # Draw the bounding box
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, LINE_THICKNESS)

                # Draw Track ID and Speed Text
                # Format speed to 1 decimal place for clarity
                label = f"ID:{track_id} {display_speed_kmh:.1f}kph"
                # Get text size to create a background rectangle
                (text_w, text_h), _ = cv2.getTextSize(label, FONT, FONT_SCALE, 1)
                # Position the text background slightly above the box
                text_bg_y1 = y1 - text_h - 5
                text_bg_y2 = y1
                # Adjust position if it goes above the frame
                if text_bg_y1 < 0:
                     text_bg_y1 = y2
                     text_bg_y2 = y2 + text_h + 5

                # Ensure text background rectangle stays within frame bounds horizontally
                text_bg_x1 = max(0, x1)
                text_bg_x2 = min(frame_width, x1 + text_w + 5)
                # Position the text within the background rectangle
                text_y_pos = text_bg_y2 - 2 if text_bg_y1 < y1 else text_bg_y1 + text_h + 2 # Adjust text position based on rect position

                # Draw text background and text only if the rectangle is valid
                if text_bg_x1 < text_bg_x2 and text_bg_y1 < text_bg_y2:
                     # Draw text background using the box color
                     cv2.rectangle(annotated_frame, (text_bg_x1, text_bg_y1), (text_bg_x2, text_bg_y2), box_color, -1)
                     # Draw the text in white
                     cv2.putText(annotated_frame, label, (text_bg_x1 + 2, text_y_pos), FONT, FONT_SCALE, TEXT_COLOR, 1, cv2.LINE_AA)


        # 4. Cleanup old tracks (remove data for objects no longer detected)
        self._cleanup_tracks(current_frame_track_ids)

        return annotated_frame

# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(description="Simple Speed Tracking with YOLO and ByteTrack")
    parser.add_argument("-i", "--input", required=True, help="Path to input video file or camera stream URL")
    parser.add_argument("-o", "--output", default=None, help="Path to save output video. If None, displays live.")
    parser.add_argument("-m", "--model", default="yolov8n.pt", help="YOLO model file")
    parser.add_argument("--conf", type=float, default=0.4, help="YOLO detection confidence threshold (default 0.4)") # Slightly higher default conf for better tracking
    parser.add_argument("--pixel-to-meter", type=float, default=0.1, help="Pixel to meter conversion ratio (default 0.05). CRITICAL for speed accuracy.")
    parser.add_argument("--video-fps", type=float, default=0, help="Video FPS (if >0, overrides auto-detection; default 0). Important for speed calculation.")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Set logging level (default INFO)")
    args = parser.parse_args()

    # Set the logging level based on the argument
    logger.setLevel(getattr(logging, args.log_level.upper()))

    logger.info("--- Simple Speed Tracking Starting ---")
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output if args.output else 'Live Display'}")
    logger.info(f"Speed Threshold: >{SPEED_THRESHOLD_KMH} kph (Box turns RED)")
    logger.info(f"Pixel-to-Meter Ratio: {args.pixel_to_meter} (Ensure this is accurate for your video)")


    # Open the video source
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        logger.error(f"Error: Could not open video source {args.input}")
        return

    # Get video properties for initializing the tracker and writer
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Use provided FPS if > 0, otherwise try to get from video, default to 30 if capture fails
    input_fps = args.video_fps if args.video_fps > 0 else cap.get(cv2.CAP_PROP_FPS)
    if input_fps <= 0:
         logger.warning("Could not determine video FPS. Defaulting to 30 FPS. Speed calculations may be inaccurate.")
         input_fps = 30.0 # Fallback if FPS cannot be determined

    logger.info(f"Input resolution: {frame_width}x{frame_height}, Effective FPS: {input_fps:.2f}")

    # Initialize the simple speed tracker processor
    tracker_processor = SimpleSpeedTracker(
        model_path=args.model,
        conf_threshold=args.conf,
        pixel_to_meter=args.pixel_to_meter,
        frame_rate=input_fps # Pass the determined effective FPS
    )

    # Setup video writer if an output path is provided
    out = None
    if args.output:
        # Define the codec (e.g., 'mp4v' for MP4). Check compatibility if needed.
        # 'mp4v' is common, 'XVID' can also work.
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        try:
             out = cv2.VideoWriter(args.output, fourcc, input_fps, (frame_width, frame_height))
             logger.info(f"Saving output video to {args.output}")
        except Exception as e:
             logger.error(f"Error setting up video writer: {e}. Output will not be saved.", exc_info=True)
             out = None # Ensure out is None if setup failed

    frame_count = 0
    start_time = time.time()

    # Main loop to process frames
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                logger.info("End of video stream or error reading frame.")
                break

            # Process the frame using the speed tracker
            annotated_frame = tracker_processor.process_frame(frame)

            # Write the annotated frame to output video if writer is setup
            if out:
                out.write(annotated_frame)

            # Display the annotated frame if no output path is given
            if args.output is None:
                cv2.imshow("Simple Speed Tracking", annotated_frame)

            frame_count += 1

            # Log progress periodically
            if frame_count % 100 == 0:
                elapsed_time = time.time() - start_time
                current_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
                logger.info(f"Processed {frame_count} frames, Current Processing FPS: {current_fps:.2f}")

            # Check for key press 'q' to exit the live display
            if args.output is None:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("User requested exit.")
                    break

    except Exception as e:
        logger.critical(f"An unexpected error occurred during processing: {e}", exc_info=True)

    finally:
        # Release video capture and writer resources
        cap.release()
        if out:
            out.release()
            logger.info(f"Output video saved to {args.output}")
        # Close all OpenCV windows if displaying live
        if args.output is None:
             cv2.destroyAllWindows()

        logger.info("--- Simple Speed Tracking Finished ---")

# Entry point of the script
if __name__ == "__main__":
    main()