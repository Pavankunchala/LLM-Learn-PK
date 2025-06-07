**Simple Speed Tracking with YOLO & ByteTrack**

A Python script for real-time vehicle detection, tracking, and speed estimation using YOLO object detection and Supervision’s ByteTrack. The system identifies vehicles, estimates their instantaneous speed (km/h), and highlights speeding vehicles with customizable threshold alerts.

---

## 🚀 Features

* **Object Detection**: Leverages YOLO for high-accuracy vehicle detection (cars, trucks, buses, motorcycles, bicycles).
* **Multi-Object Tracking**: Integrates ByteTrack for persistent tracking across frames with unique IDs.
* **Speed Estimation**: Calculates instantaneous speed in km/h using pixel-to-meter conversion and timestamped trajectories.
* **Smoothing**: Applies exponential moving average for stable speed readings.
* **Visual Alerts**: Draws bounding boxes in green (normal) or red (speeding) with ID and speed label.
* **Logging**: Built-in logging to monitor processing status, errors, and performance metrics.
* **Extensible**: Configurable detection confidence, speed thresholds, pixel-to-meter ratios, and video sources.

---

## 📋 Prerequisites

* Python 3.8+
* OpenCV
* NumPy
* Ultralytics YOLO (`ultralytics` package)
* Supervision (`supervision` package)

---

## 🔧 Installation

1. **Clone the repository**

```bash
# 1. Create a new folder and init
mkdir speed-tracking
cd speed-tracking
git init

# 2. Add the remote and enable sparse checkout
git remote add origin https://github.com/Pavankunchala/LLM-Learn-PK.git
git config core.sparseCheckout true

# 3. Specify the folder you want
echo "speed-tracking/*" > .git/info/sparse-checkout

# 4. Pull just that folder
git pull origin main

   ```

2. **Create a virtual environment (optional but recommended)**

   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate     # Windows
   ```

3. **Install Python dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Download YOLO model weights** (if not included)

   ```bash
   # Example: YOLOv8n
   # The script defaults to "yolov8n.pt". Place the file in this directory or specify a path.
   ```

---

## ⚙️ Usage

Run the script from the command line with your video source:

```bash
python speed_tracker.py \
  --input path/to/video.mp4 \
  --model yolov8n.pt \
  --conf 0.4 \
  --pixel-to-meter 0.1 \
  --video-fps 30 \
  --output output.mp4 \
  --log-level INFO
```

| Option             | Description                                                 | Default      |
| ------------------ | ----------------------------------------------------------- | ------------ |
| `-i, --input`      | Path to input video file or camera URL (required)           | —            |
| `-o, --output`     | Path to save annotated output video (live display if none)  | None         |
| `-m, --model`      | YOLO model file (weights)                                   | `yolov8n.pt` |
| `--conf`           | YOLO detection confidence threshold                         | `0.4`        |
| `--pixel-to-meter` | Pixel-to-meter conversion factor                            | `0.1`        |
| `--video-fps`      | Video FPS override; if ≤0, auto-detected or defaulted to 30 | `0`          |
| `--log-level`      | Logging verbosity (`DEBUG`/`INFO`/`WARNING`/`ERROR`)        | `INFO`       |

### Live Display

Omit `--output` to see a live window. Press `q` to exit.

---

## 📐 Configuration Tips

* **Pixel-to-Meter Ratio**: Calibrate by measuring a known distance in pixels (e.g., lane width).
* **Speed Threshold**: Adjust `SPEED_THRESHOLD_KMH` in script for local speed limits.
* **Trajectory Length**: Modify `MAX_TRAJECTORY_LENGTH` to balance responsiveness vs. stability.
* **Smoothing Factor**: Tweak `SPEED_SMOOTHING_FACTOR` (0–1) to control speed read smoothing.

---

## 🛠 Troubleshooting

* **No Detections**: Check model weights and increase `--conf` if objects are missed.
* **FPS Warning**: If video FPS is unknown, specify `--video-fps` to ensure accurate speed.
* **Video Writer Errors**: Ensure correct codec support; try `XVID` or `MJPG` if `mp4v` fails.

---

## 📈 Performance

* Processes \~15–25 FPS on an NVIDIA GPU (varies by resolution/model).
* About 20% faster processing with YOLOv8n vs. YOLOv8s in initial tests.

---

## 🤝 Contribution & Support

Contributions welcome! For issues or feature requests, please open an issue or submit a pull request.

---

## 📄 License

MIT License © 2025 Your Name
