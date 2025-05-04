# Polygon Object Tracker

A full-stack application for real-time object detection and counting within user-defined polygonal zones on video feeds, featuring an interactive web GUI.

## Project Goal

To provide an integrated web interface for defining detection zones (polygons) on a video stream (upload or webcam), selecting target object classes, setting detection parameters, and viewing real-time annotated video with object counts per zone. This replaces manual configuration via files with an interactive GUI.

## Features

*   **Web-Based GUI:** Interactive frontend built with React and TypeScript.
*   **Video Input:** Supports processing uploaded video files or live webcam feeds.
*   **Interactive Polygon Drawing:** Users can draw multiple polygonal zones directly on the video preview.
*   **Dynamic Configuration:** Select specific object classes (e.g., person, car, bicycle) and adjust the detection confidence threshold via the UI.
*   **Real-time Processing:** Utilizes YOLOv8 and Supervision for efficient object detection within defined zones.
*   **Live Annotated Feed:** Streams the processed video frames (with bounding boxes and zone annotations) back to the frontend via WebSockets.
*   **Per-Zone Counts:** Displays real-time counts of detected objects within each defined zone.
*   **Processed Video Output:** Saves the fully processed video with annotations to a server-side folder upon completion.
*   **Final Video Playback:** Allows viewing the saved, processed video directly in the UI after processing finishes.

## Tech Stack

*   **Backend:** Python, asyncio, YOLOv8 (Ultralytics), Supervision, WebSockets, aiohttp (for uploads)
*   **Frontend:** React, TypeScript, Zustand (State Management), Tailwind CSS (Styling), Axios (HTTP requests)
*   **Communication:** WebSockets (Real-time data), HTTP (File uploads, Serving final video)

## Setup and Installation

### Prerequisites

*   Python 3.9+
*   pip (Python package installer)
*   Node.js 16+
*   npm or yarn (Node package manager)

### Backend Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-name>/backend
    ```
2.  **Create and activate a virtual environment (Recommended):**
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```
3.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: Depending on your system and CUDA setup, you might need a specific version of `torch`. Consult the PyTorch and Ultralytics documentation if you encounter installation issues.*
4.  **Download a YOLOv8 model:** Download a model file (e.g., `yolov8m.pt`) from the Ultralytics repository and place it in the `backend` directory, or be prepared to specify its path via the command line argument.

### Frontend Setup

1.  **Navigate to the frontend directory:**
    ```bash
    cd ../frontend
    ```
    *(Assuming you are in the `backend` directory from the previous step)*
2.  **Install Node dependencies:**
    ```bash
    npm install
    # OR
    yarn install
    ```
3.  **Configure Environment Variables (Optional):**
    *   If your backend server runs on a different host or port than the defaults (`localhost:8000` for HTTP, `localhost:8765` for WS), create a `.env` file in the `frontend` directory:
        ```env
        # frontend/.env
        VITE_BACKEND_HTTP_HOST=your_backend_host
        VITE_BACKEND_HTTP_PORT=your_backend_http_port
        VITE_BACKEND_WS_HOST=your_backend_ws_host
        VITE_BACKEND_WS_PORT=your_backend_ws_port
        ```

## Usage

1.  **Start the Backend Server:**
    *   Navigate to the `backend` directory.
    *   Activate your virtual environment if you created one.
    *   Run `main.py`, providing the path to your YOLO model:
        ```bash
        python main.py --model yolov8m.pt
        # Or specify a different path:
        # python main.py --model /path/to/your/yolov8m.pt
        ```
    *   The server will print messages indicating the HTTP and WebSocket servers are running.

2.  **Start the Frontend Development Server:**
    *   Navigate to the `frontend` directory.
    *   Run the development server:
        ```bash
        npm run dev
        # OR
        yarn dev
        ```
    *   The command will output the URL where the frontend is accessible (usually `http://localhost:5173` or similar).

3.  **Access the Application:**
    *   Open the frontend URL in your web browser.

4.  **Workflow:**
    *   Select a video source (Upload a file or use Webcam).
    *   If using a file, wait for the local preview to load.
    *   Draw one or more polygonal zones on the video preview area.
    *   Use the control panel to select the object classes you want to detect/count.
    *   Adjust the confidence threshold slider.
    *   (Optional, for file uploads) Use the playback controls to navigate the video and click "Set Start Time" to begin processing from a specific point.
    *   Click "Start Processing".
    *   View the real-time annotated video feed and the detection counts per zone in the results panel.
    *   Processing will stop automatically when the video ends or if you click "Stop Processing".
    *   If processing a file completed successfully, the display will switch to show the final processed video saved on the server. You can use the browser controls to play it or download it using the provided button. Click the 'X' button to close the final video view and return to the preview/initial state.

## Configuration

*   **Backend:** Server ports, upload/output folder paths, allowed origins (CORS), and max file size can be adjusted in `backend/config.py`.
*   **Frontend:** Backend connection URLs can be set via environment variables in `frontend/.env` (see Setup).