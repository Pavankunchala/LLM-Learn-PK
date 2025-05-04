// components/ControlPanel.tsx

import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Play, Square, Settings, Filter, Target, CircleDot, Wifi, XCircle, Loader2, Camera, Upload, Info } from 'lucide-react';
import { useDetectionStore } from '../store/detectionStore';
import ClassSelector from './ClassSelector';
import ConfidenceSlider from './ConfidenceSlider';
import { AVAILABLE_CLASSES } from '../constants/classNames'; // Assuming AVAILABLE_CLASSES is still used
import WebSocketManager from '../utils/websocketManager'; // Import WebSocketManager

const ControlPanel = () => {
  const [expanded, setExpanded] = useState({
    classes: true,
    configuration: true,
  });

  const {
    isProcessing,
    startProcessing, // Store action to update frontend state
    stopProcessing,  // Store action to update frontend state
    selectedClasses,
    polygons,
    confidenceThreshold,
    videoSource,
    uploadedFilePath, // Get the backend file path from here
    uploadedFileName, // Get the file name for display
    startFrameTime, // Get the selected start time from the store
    backendStatus, // Get backend connection status
    backendStatusMessage // Get backend status message
  } = useDetectionStore();

  const toggleSection = (section: 'classes' | 'configuration') => {
    setExpanded({
      ...expanded,
      [section]: !expanded[section]
    });
  };

  const handleStartProcessing = () => {
    if (isStartButtonDisabled()) return;

    // Determine the video path/index to send to backend
    let videoPath: string | number | null = null;
    if (videoSource === 'webcam') {
      videoPath = 0; // Use index 0 for webcam on the backend
    } else if (videoSource === 'file') {
      videoPath = uploadedFilePath; // Use the temporary path from backend upload
    }

    if (videoPath === null) {
         console.error("Cannot start processing: No valid video source path/index determined.");
         // Optionally show a user-facing error message via state
         // useDetectionStore.getState().setBackendStatus("error", "No video source selected or file not uploaded.");
         return;
    }

    // Collect configuration from the store
    const config: any = { // Use 'any' or a more specific type if needed for flexibility
      polygons: polygons, // Array of {id: string, points: {x, y}[]}
      selectedClasses: selectedClasses, // Array of string names
      confidenceThreshold: confidenceThreshold, // Float
      videoSource: videoSource, // 'webcam' or 'file'
      videoPath: videoPath // The path string or 0
    };

    // Add startFrameTime ONLY if the source is a file
    if (videoSource === 'file') {
        config.startFrameTime = startFrameTime;
    }


    console.log("Sending configuration:", config);
    // Send configuration via WebSocket
    WebSocketManager.sendConfiguration(config);

    // Send control message to start processing
    console.log("Sending start control message.");
    WebSocketManager.sendControl('start');

    // The isProcessing state will be updated by the WebSocketManager
    // when it receives the 'started' status message from the backend.
    // We don't call startProcessing() here directly anymore.
  };

   const handleStopProcessing = () => {
        if (!isProcessing) return;

        console.log("Sending stop control message.");
        // Send control message to stop processing
        WebSocketManager.sendControl('stop');

        // The isProcessing state will be updated by the WebSocketManager
        // when it receives the 'stopped' status message from the backend.
        // We don't call stopProcessing() here directly anymore.
   };


  const isStartButtonDisabled = () => {
    const backendReady = backendStatus === 'connected' || backendStatus === 'config_received';
    const sourceSelected = videoSource !== null;
    const sourcePathAvailable = videoSource === 'webcam' || (videoSource === 'file' && uploadedFilePath !== null);
    const hasClasses = selectedClasses.length > 0;
    const hasPolygons = polygons.length > 0;
    // For file source, require startFrameTime to be set (or just accept 0 as default)
    // Let's not strictly require it, 0 is a valid start time.
    // const isStartFrameSetForFile = videoSource !== 'file' || startFrameTime !== null;


    // Check if backend is in a state where starting is possible (not already started or stopping)
    const backendCanStart = backendStatus !== 'started' && backendStatus !== 'stopping' && backendStatus !== 'connecting'; // Cannot start if connecting

    return (
      isProcessing || // Disable if frontend UI state is already processing
      !backendReady || // Disable if backend is not connected/ready
      !backendCanStart || // Disable if backend is in a state where it can't start
      !sourceSelected || // Disable if no video source type is chosen
      !sourcePathAvailable || // Disable if file source chosen but no file uploaded yet
      !hasClasses || // Disable if no classes selected
      !hasPolygons // Disable if no polygons drawn
      // || !isStartFrameSetForFile // Apply start frame requirement if needed
    );
  };

  // Check if the stop button should be disabled
  const isStopButtonDisabled = () => {
      // Button is enabled only when backend status is 'started' or 'stopping'
      return backendStatus !== 'started' && backendStatus !== 'stopping';
  };


  return (
    <div className="bg-gray-800 rounded-xl shadow-lg overflow-hidden">
      {/* Header */}
      <div className="p-4 bg-gray-700 flex justify-between items-center">
        <h2 className="text-lg font-medium flex items-center">
          <Settings size={18} className="mr-2" />
          Control Panel
        </h2>
         {/* Backend Status Indicator */}
         <div className="flex items-center text-sm">
             {backendStatus === 'connected' || backendStatus === 'config_received' || backendStatus === 'started' || backendStatus === 'stopping' ? (
                  <Wifi size={16} className="mr-1 text-success-500" />
             ) : backendStatus === 'connecting' ? (
                  <Loader2 size={16} className="mr-1 text-primary-400 animate-spin" />
             ) : (
                  <XCircle size={16} className="mr-1 text-error-500" />
             )}
             <span className={`font-medium capitalize ${
                  backendStatus === 'connected' || backendStatus === 'config_received' || backendStatus === 'started' || backendStatus === 'stopping' ? 'text-success-400' :
                  backendStatus === 'connecting' ? 'text-primary-400' :
                  backendStatus === 'warning' ? 'text-warning-400' :
                  'text-error-400'
             }`}>
                 {backendStatus}
             </span>
         </div>
      </div>

      {/* Control Sections */}
      <div className="p-4 space-y-4">
        {/* Source Info */}
        <div className="border border-gray-700 rounded-lg p-3">
             <p className="text-sm text-gray-400 mb-2">Video Source</p>
             <div className="bg-gray-700 rounded-lg p-2 text-center flex items-center justify-center">
                  {videoSource === 'webcam' ? (
                       <p className="text-sm flex items-center"><Camera size={14} className="mr-1"/> Webcam</p>
                  ) : videoSource === 'file' && uploadedFileName ? (
                       <p className="text-sm flex items-center"><Upload size={14} className="mr-1"/> {uploadedFileName}</p>
                  ) : (
                       <p className="text-sm text-gray-400">No source selected/uploaded</p>
                  )}
             </div>
             {/* Display start time for file source */}
             {videoSource === 'file' && uploadedFileName && (
                  <p className="text-xs text-gray-500 mt-1">
                      Start time for processing: {startFrameTime.toFixed(2)} s
                  </p>
             )}
        </div>

        {/* Classes Section */}
        <div className="border border-gray-700 rounded-lg overflow-hidden">
          <button
            className="w-full flex items-center justify-between p-3 bg-gray-700 hover:bg-gray-600 transition-colors"
            onClick={() => toggleSection('classes')}
          >
            <div className="flex items-center">
              <Filter size={16} className="mr-2" />
              <span className="font-medium">Detection Classes</span>
            </div>
            <span className="text-sm text-gray-400">
              {selectedClasses.length} selected
            </span>
          </button>

          {expanded.classes && (
            <div className="p-3">
              <ClassSelector classes={AVAILABLE_CLASSES} /> {/* Assuming AVAILABLE_CLASSES is used */}
            </div>
          )}
        </div>

        {/* Configuration Section */}
        <div className="border border-gray-700 rounded-lg overflow-hidden">
          <button
            className="w-full flex items-center justify-between p-3 bg-gray-700 hover:bg-gray-600 transition-colors"
            onClick={() => toggleSection('configuration')}
          >
            <div className="flex items-center">
              <Target size={16} className="mr-2" />
              <span className="font-medium">Configuration</span>
            </div>
          </button>

          {expanded.configuration && (
            <div className="p-3 space-y-4">
              <div>
                <ConfidenceSlider />
              </div>

              <div>
                <p className="text-sm text-gray-400 mb-2">Detection Zones</p>
                <div className="bg-gray-700 rounded-lg p-2 text-center flex items-center justify-center">
                   <CircleDot size={14} className="mr-1 text-blue-400"/>
                  {polygons.length === 0 ? (
                    <p className="text-gray-400 text-sm">No zones defined</p>
                  ) : (
                    <p className="text-sm">
                      {polygons.length} zone{polygons.length !== 1 ? 's' : ''} defined
                    </p>
                  )}
                </div>
                <p className="text-xs text-gray-500 mt-1">
                  Draw zones on the video panel
                </p>
              </div>
            </div>
          )}
        </div>

        {/* Backend Status Message */}
        {backendStatusMessage && (backendStatus === 'error' || backendStatus === 'warning' || backendStatus === 'disconnected' || backendStatus === 'connecting') && (
             <div className={`p-3 rounded-lg text-sm flex items-start ${backendStatus === 'error' ? 'bg-error-900 text-error-300' : backendStatus === 'warning' ? 'bg-warning-900 text-warning-300' : backendStatus === 'connecting' ? 'bg-primary-900 text-primary-300' : 'bg-gray-700 text-gray-300'}`}>
                 {backendStatus === 'error' || backendStatus === 'disconnected' ? <XCircle size={16} className="mr-2 flex-shrink-0 mt-0.5" /> : backendStatus === 'connecting' ? <Loader2 size={16} className="mr-2 flex-shrink-0 mt-0.5 animate-spin" /> : <Info size={16} className="mr-2 flex-shrink-0 mt-0.5" />} {/* Use Info for warning */}
                 <p>{backendStatusMessage}</p>
             </div>
        )}


        {/* Control Buttons */}
        <div className="grid grid-cols-2 gap-2 pt-2">
          {/* Start Button */}
            <motion.button
              whileHover={{ scale: isStartButtonDisabled() ? 1 : 1.03 }}
              whileTap={{ scale: isStartButtonDisabled() ? 1 : 0.97 }}
              className={`col-span-2 py-3 rounded-lg flex items-center justify-center transition-colors ${
                isStartButtonDisabled()
                  ? 'bg-gray-700 text-gray-500 cursor-not-allowed'
                  : 'bg-primary-600 hover:bg-primary-500 text-white'
              }`}
              onClick={handleStartProcessing}
              disabled={isStartButtonDisabled()}
            >
              <Play size={18} className="mr-2" />
              <span>Start Detection</span>
            </motion.button>

          {/* Stop Button */}
            <motion.button
              whileHover={{ scale: isStopButtonDisabled() ? 1 : 1.03 }}
              whileTap={{ scale: isStopButtonDisabled() ? 1 : 0.97 }}
              className={`col-span-2 py-3 rounded-lg flex items-center justify-center transition-colors ${
                 isStopButtonDisabled() ? 'bg-gray-700 text-gray-500 cursor-not-allowed' : 'bg-error-600 hover:bg-error-500 text-white'
              }`}
              onClick={handleStopProcessing}
               disabled={isStopButtonDisabled()}
            >
              <Square size={18} className="mr-2" />
              <span>Stop Detection</span>
            </motion.button>
        </div>
      </div>
    </div>
  );
};

export default ControlPanel;