// components/ConfidenceSlider.tsx
// (No changes needed)

import React from 'react';
import { useDetectionStore } from '../store/detectionStore';

const ConfidenceSlider = () => {
  const { confidenceThreshold, setConfidenceThreshold } = useDetectionStore();

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setConfidenceThreshold(parseFloat(e.target.value));
  };

  return (
    <div>
      <div className="flex justify-between mb-1">
        <label htmlFor="confidence-slider" className="text-sm text-gray-400">
          Confidence Threshold
        </label>
        <span className="text-sm font-medium text-gray-300"> {/* Adjusted text color */}
          {(confidenceThreshold * 100).toFixed(0)}%
        </span>
      </div>

      <input
        id="confidence-slider"
        type="range"
        min="0"
        max="1"
        step="0.05"
        value={confidenceThreshold}
        onChange={handleChange}
        className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-primary-500"
      />

      <div className="flex justify-between text-xs text-gray-500 mt-1 px-1"> {/* Added some padding */}
        <span>0%</span>
        <span>50%</span>
        <span>100%</span>
      </div>
    </div>
  );
};

export default ConfidenceSlider;