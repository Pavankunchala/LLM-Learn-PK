import React from 'react';
import './ProcessingOverlay.css'; // Create this CSS file for styles

const ProcessingOverlay = ({ isProcessing }) => {
  if (!isProcessing) return null;

  return (
    <div className="processing-overlay">
      <div className="processing-container">
        <div className="processing-spinner"></div>
        <h3 className="processing-text">Tailoring your resume...</h3>
        <div className="progress-bar">
          <div className="progress-bar-inner"></div>
        </div>
      </div>
    </div>
  );
};

export default ProcessingOverlay;
