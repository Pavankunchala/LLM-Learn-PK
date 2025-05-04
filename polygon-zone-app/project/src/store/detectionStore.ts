// store/detectionStore.ts

import { create } from 'zustand';
import { Polygon } from '../types/detection';

interface DetectionState {
  // Video source control
  videoSource: 'webcam' | 'file' | null;
  setVideoSource: (source: 'webcam' | 'file' | null) => void;
  uploadedFilePath: string | number | null; // Backend path (string) or webcam index (number)
  setUploadedFilePath: (path: string | number | null) => void;
  uploadedFileName: string | null;
  setUploadedFileName: (name: string | null) => void;
  startFrameTime: number;
  setStartFrameTime: (time: number) => void;

  // Polygon management
  polygons: Polygon[];
  addPolygon: (polygon: Polygon) => void;
  removeLastPolygon: () => void;
  clearPolygons: () => void;

  // Detection classes
  selectedClasses: string[];
  toggleClass: (className: string) => void;

  // Configuration
  confidenceThreshold: number;
  setConfidenceThreshold: (value: number) => void;

  // Backend Connection & Processing State
  backendStatus: 'disconnected' | 'connecting' | 'connected' | 'config_received' | 'started' | 'stopping' | 'stopped' | 'error' | 'warning';
  backendStatusMessage: string | null;
  // MODIFIED: Accept potential extra data in status payload
  setBackendStatus: (status: DetectionState['backendStatus'], message?: string | null, extraData?: Record<string, any> | null) => void;

  isProcessing: boolean;
  startProcessing: () => void;
  stopProcessing: () => void;

  // Results
  processedFrameUrl: string | null; // Live frame Base64
  setProcessedFrameUrl: (url: string | null) => void;

  // Final Output File (NEW STATE)
  finalOutputFilename: string | null;
  setFinalOutputFilename: (name: string | null) => void; // Explicit setter, though setBackendStatus handles it

  // Detection results
  detectionResults: Record<string, Record<string, number>>;
  updateDetectionResults: (results: Record<string, Record<string, number>>) => void;
}

export const useDetectionStore = create<DetectionState>((set, get) => ({
  // Video source control
  videoSource: null,
  setVideoSource: (source) => set({ videoSource: source, uploadedFilePath: null, uploadedFileName: null, startFrameTime: 0, finalOutputFilename: null }), // Clear final output on source change
  uploadedFilePath: null,
  setUploadedFilePath: (path) => set({ uploadedFilePath: path }),
  uploadedFileName: null,
  setUploadedFileName: (name) => set({ uploadedFileName: name }),
  startFrameTime: 0,
  setStartFrameTime: (time) => set({ startFrameTime: time }),

  // Polygons
  polygons: [],
  addPolygon: (polygon) => set((state) => ({ polygons: [...state.polygons, polygon] })),
  removeLastPolygon: () => set((state) => ({ polygons: state.polygons.slice(0, -1) })),
  clearPolygons: () => set({ polygons: [] }),

  // Classes
  selectedClasses: [],
  toggleClass: (className) => set((state) => ({ selectedClasses: state.selectedClasses.includes(className) ? state.selectedClasses.filter(name => name !== className) : [...state.selectedClasses, className] })),

  // Config
  confidenceThreshold: 0.4,
  setConfidenceThreshold: (value) => set({ confidenceThreshold: value }),

  // Backend Status (MODIFIED)
  backendStatus: "disconnected",
  backendStatusMessage: null,
  setBackendStatus: (status, message = null, extraData = null) => {
      set({ backendStatus: status, backendStatusMessage: message });
      // If stopped successfully AND we received an output filename, store it
      if (status === 'stopped' && extraData?.outputFilename) {
          console.log(`Received final output filename: ${extraData.outputFilename}`);
          set({ finalOutputFilename: extraData.outputFilename });
      }
      // Optionally clear filename if status changes to something else?
      // stopProcessing handles clearing it when explicitly stopped/errored.
  },

  // Processing State
  isProcessing: false,
  startProcessing: () => set({ isProcessing: true, finalOutputFilename: null }), // Clear final output when starting
  stopProcessing: () => set({ // Clear final output on stop/error too
    isProcessing: false,
    processedFrameUrl: null,
    detectionResults: {},
    finalOutputFilename: null
  }),

  // Results
  processedFrameUrl: null,
  setProcessedFrameUrl: (url) => set({ processedFrameUrl: url }),

  // Final Output File (NEW STATE & SETTER)
  finalOutputFilename: null,
  setFinalOutputFilename: (name) => set({ finalOutputFilename: name }), // Direct setter if needed

  // Detections
  detectionResults: {},
  updateDetectionResults: (results) => set({ detectionResults: results }),
}));