// App.tsx

import React, { useEffect } from 'react'; // Import useEffect
import { motion } from 'framer-motion';
import AppHeader from './components/AppHeader';
import VideoPanel from './components/VideoPanel';
import ControlPanel from './components/ControlPanel';
import ResultsPanel from './components/ResultsPanel';
import { useDetectionStore } from './store/detectionStore';
import WebSocketManager from './utils/websocketManager'; // Import WebSocketManager

function App() {
  const isProcessing = useDetectionStore((state) => state.isProcessing);

  // Connect to WebSocket when the app mounts
  useEffect(() => {
    console.log("App mounted. Connecting WebSocket...");
    WebSocketManager.connect();

    // Optional: Add cleanup for WebSocket on unmount
    // However, WebSocketManager is a singleton and might persist.
    // Disconnecting here might interfere with hot-reloading in development.
    // For a production app, you might manage this differently (e.g., in a dedicated hook)
    return () => {
      console.log("App unmounting. Disconnecting WebSocket...");
      // WebSocketManager.disconnect(); // Commented out for development ease
    };
  }, []); // Empty dependency array ensures this runs only once on mount

  return (
    <div className="flex flex-col min-h-screen bg-gray-900 text-white">
      <AppHeader />
      <main className="flex flex-col lg:flex-row flex-1 p-4 gap-4 max-w-screen-2xl mx-auto w-full overflow-hidden">
        <motion.div
          className="flex-1 min-w-0 flex flex-col" // Use flex-col to make VideoPanel fill height
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          <VideoPanel />
        </motion.div>

        <motion.div
          className="w-full lg:w-80 flex flex-col gap-4" // Fixed width for control/results panel
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
        >
          <ControlPanel />
          {isProcessing && <ResultsPanel />} {/* Show results panel only when processing */}
        </motion.div>
      </main>
    </div>
  );
}

export default App;