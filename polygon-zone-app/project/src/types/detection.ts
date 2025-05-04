// types/detection.ts

// Represents a point in normalized coordinates (0-1)
export interface Point {
  x: number; // 0.0 to 1.0
  y: number; // 0.0 to 1.0
}

// Represents a polygon zone defined by normalized points
export interface Polygon {
  id: string; // Unique identifier for the polygon (e.g., timestamp string)
  points: Point[];
}

// --- WebSocket Message Types (Frontend -> Backend) ---

// Configuration message sent when starting detection
export interface ConfigurationMessage {
  type: 'configuration';
  polygons: Polygon[];
  selectedClasses: string[]; // Array of class name strings (e.g., ["person", "car"])
  confidenceThreshold: number; // Float between 0 and 1
  videoSource: 'webcam' | 'file';
  videoPath: string | number; // Path string for file, 0 for webcam index
  startFrameTime?: number; // Optional: Start time in seconds for file sources
}

// Control message to start or stop processing
export interface ControlMessage {
  type: 'control';
  action: 'start' | 'stop' | 'pause' | 'resume'; // Backend currently only supports start/stop
}

// --- WebSocket Message Types (Backend -> Frontend) ---

// Message containing a processed video frame as a base64 image URL
export interface ProcessedFrameMessage {
  type: 'processedFrame';
  imageUrl: string; // Data URL (e.g., "data:image/jpeg;base64,...")
}

// Message containing updated detection counts per zone and class
export interface ResultsUpdateMessage {
  type: 'resultsUpdate';
  results: Record<string, Record<string, number>>; // { zone_id: { class_id_string: count, ... }, ... }
}

// Message indicating backend status (connection, processing state, errors)
export interface StatusMessage {
  type: 'status';
  status: 'disconnected' | 'connecting' | 'connected' | 'config_received' | 'started' | 'stopping' | 'stopped' | 'error' | 'warning';
  message?: string; // Optional human-readable message
}

// Union type for all backend messages the frontend expects
export type BackendMessage = ProcessedFrameMessage | ResultsUpdateMessage | StatusMessage;