// components/ResultsPanel.tsx
// (No changes needed, assumes backend sends data in the expected format)

import React from 'react';
import { motion } from 'framer-motion';
import { BarChart, Clock, Users, Info } from 'lucide-react'; // Added Info icon
import { useDetectionStore } from '../store/detectionStore';
import { AVAILABLE_CLASSES } from '../constants/classNames'; // Still using frontend constant for display name lookup

const ResultsPanel = () => {
  const { detectionResults, isProcessing } = useDetectionStore();

  // Only render this panel if processing is active and there are results
  if (!isProcessing) return null;

  const totalObjects = Object.values(detectionResults).reduce((total, zone) => {
      return total + Object.values(zone).reduce((sum, count) => sum + count, 0);
  }, 0);

  const hasResults = Object.keys(detectionResults).length > 0 && totalObjects > 0;


  return (
    <motion.div
      className="bg-gray-800 rounded-xl shadow-lg overflow-hidden"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
    >
      <div className="p-4 bg-gray-700">
        <h2 className="text-lg font-medium flex items-center">
          <BarChart size={18} className="mr-2" />
          Detection Results
        </h2>
      </div>

      <div className="p-4">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center">
            <Clock size={16} className="mr-1 text-gray-400" />
            <span className="text-sm text-gray-400">Live tracking</span>
          </div>

          <div className="flex items-center">
            <Users size={16} className="mr-1 text-gray-400" />
            <span className="text-sm font-medium">
              {totalObjects} object{totalObjects !== 1 ? 's' : ''}
            </span>
          </div>
        </div>

        <div className="space-y-4">
          {hasResults ? (
            // Display results per zone
            Object.entries(detectionResults).map(([zoneId, counts]) => (
              // Filter out zones with zero counts if desired, currently shows all zones sent by backend
              // if (Object.values(counts).reduce((sum, count) => sum + count, 0) === 0) return null;

              <div key={zoneId} className="border border-gray-700 rounded-lg p-3">
                <div className="flex justify-between items-center mb-2">
                  <h3 className="font-medium">{zoneId}</h3> {/* Use the zoneId string sent by backend */}
                  <span className="text-sm text-gray-400">
                    {Object.values(counts).reduce((sum, count) => sum + count, 0)} total
                  </span>
                </div>

                <div className="space-y-1.5">
                  {/* Display counts per class within the zone */}
                  {Object.entries(counts).map(([classId, count]) => (
                    // Only show classes that have a count > 0 in this zone
                    // if (count === 0) return null;

                    <div key={classId} className="flex justify-between items-center">
                      <div className="flex items-center">
                        <div className="w-2 h-2 rounded-full bg-primary-500 mr-2"></div> {/* Generic indicator color */}
                        {/* Use frontend constant for display name lookup */}
                        <span className="text-sm">{AVAILABLE_CLASSES[classId] || classId}</span>
                      </div>
                      <span className="font-medium">{count}</span>
                    </div>
                  ))}
                </div>
              </div>
            ))
          ) : (
            // Message when processing starts but no detections found yet
            <div className="text-center py-4 text-gray-500 flex flex-col items-center">
              <Info size={24} className="mb-2"/>
              <p>No detections found yet</p>
              <p className="text-xs mt-1">Waiting for objects to appear in zones...</p>
            </div>
          )}
        </div>
      </div>
    </motion.div>
  );
};

export default ResultsPanel;