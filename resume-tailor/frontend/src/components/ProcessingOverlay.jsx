import React, { useState, useEffect } from 'react';
import './ProcessingOverlay.css';

const ProcessingOverlay = ({ isProcessing, progress = 0, status = '', onComplete = null }) => {
  const [isComplete, setIsComplete] = useState(false);
  const [animationComplete, setAnimationComplete] = useState(false);
  const [stageIndex, setStageIndex] = useState(0);
  
  // Define the animation stages with their respective progress ranges
  const stages = [
    { min: 0, max: 10, text: 'Analyzing resume content...', icon: '📄' },
    { min: 10, max: 30, text: 'Identifying key skills and experiences...', icon: '🔍' },
    { min: 30, max: 50, text: 'Comparing with job description...', icon: '📊' },
    { min: 50, max: 70, text: 'Tailoring content to match requirements...', icon: '✏️' },
    { min: 70, max: 90, text: 'Formatting and generating PDF...', icon: '📝' },
    { min: 90, max: 100, text: 'Finalizing document...', icon: '✨' }
  ];
  
  // Update the current stage based on progress
  useEffect(() => {
    const newStageIndex = stages.findIndex(
      stage => progress >= stage.min && progress <= stage.max
    );
    
    if (newStageIndex !== -1 && newStageIndex !== stageIndex) {
      setStageIndex(newStageIndex);
    }
  }, [progress, stageIndex]);
  
  // Handle completion
  useEffect(() => {
    // If the process is complete (100%), wait a moment before hiding
    if (progress >= 100 && (status === 'completed' || status === 'error')) {
      setIsComplete(true);
      
      const timer = setTimeout(() => {
        setAnimationComplete(true);
        if (onComplete) onComplete();
      }, 1500); // Give a short delay to show the 100% status
      
      return () => clearTimeout(timer);
    }
  }, [progress, status, onComplete]);
  
  // Don't render if not processing or animation has completed
  if (!isProcessing || animationComplete) return null;
  
  // Calculate the progress bar fill width
  const progressWidth = `${progress}%`;
  
  // Get the stage message, either from the backend status or from the stages array
  const stageMessage = status || (stageIndex >= 0 && stageIndex < stages.length 
    ? stages[stageIndex].text 
    : 'Processing...');
    
  // Get the current stage icon
  const stageIcon = stageIndex >= 0 && stageIndex < stages.length 
    ? stages[stageIndex].icon 
    : '🔄';
  
  // Determine status class
  let statusClass = 'processing';
  let statusTitle = 'Tailoring Your Resume';
  
  if (status === 'error' || status === 'error') {
    statusClass = 'error';
    statusTitle = 'Error Processing Resume';
  } else if (isComplete) {
    statusClass = 'complete';
    statusTitle = 'Tailoring Complete!';
  }

  return (
    <div className={`processing-overlay ${statusClass}`}>
      <div className="processing-modal">
        <div className="processing-icon">
          <div className="icon-animation">
            <div className="document original"></div>
            <div className="document-shadow"></div>
            <div className="processing-rays">
              <div className="ray ray-1"></div>
              <div className="ray ray-2"></div>
              <div className="ray ray-3"></div>
              <div className="ray ray-4"></div>
              <div className="ray ray-5"></div>
            </div>
            <div className="document tailored"></div>
          </div>
        </div>
        
        <h3 className="processing-title">{statusTitle}</h3>
        
        <div className="stage-indicator">
          <div className="stage-icon">{stageIcon}</div>
          <p className="processing-message">
            {stageMessage}
          </p>
        </div>
        
        <div className="progress-container">
          <div className="progress-bar">
            <div 
              className={`progress-fill ${isComplete ? 'complete' : ''}`} 
              style={{ width: progressWidth }}
            ></div>
          </div>
          <div className="progress-label">{Math.round(progress)}%</div>
        </div>
        
        <div className="processing-footer">
          <span className="ai-model-label">Using model:</span>
          <span className="ai-model-name">{window.selectedModel || 'AI'}</span>
        </div>
        
        {status === 'error' && (
          <div className="error-details">
            <p className="error-message">
              An error occurred during processing. The system will try to recover and provide a basic resume.
            </p>
          </div>
        )}
        
        {isComplete && (
          <div className="completion-message">
            {status === 'error' ? (
              <p>We encountered some issues, but created a basic resume for you.</p>
            ) : (
              <p>Your tailored resume is ready!</p>
            )}
          </div>
        )}
      </div>
      
      <style jsx>{`
        .processing-overlay {
          position: fixed;
          top: 0;
          left: 0;
          width: 100%;
          height: 100%;
          background-color: rgba(15, 23, 42, 0.8);
          display: flex;
          align-items: center;
          justify-content: center;
          z-index: 1000;
          backdrop-filter: blur(4px);
          animation: fade-in 0.3s ease;
        }
        
        .processing-modal {
          background-color: white;
          border-radius: 1rem;
          box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2);
          padding: 2rem;
          width: 90%;
          max-width: 500px;
          overflow: hidden;
          animation: slide-up 0.4s ease;
        }
        
        .processing-icon {
          width: 120px;
          height: 120px;
          margin: 0 auto 1.5rem;
          position: relative;
        }
        
        .icon-animation {
          width: 100%;
          height: 100%;
          position: relative;
        }
        
        .document {
          position: absolute;
          width: 60px;
          height: 80px;
          background-color: #f1f5f9;
          border-radius: 4px;
          border: 2px solid #94a3b8;
          top: 50%;
          transform: translateY(-50%);
        }
        
        .document:before, .document:after {
          content: '';
          position: absolute;
          background-color: #94a3b8;
          height: 2px;
        }
        
        .document:before {
          width: 70%;
          top: 20px;
          left: 15%;
        }
        
        .document:after {
          width: 50%;
          top: 30px;
          left: 15%;
        }
        
        .document.original {
          left: 10px;
          z-index: 1;
        }
        
        .document.tailored {
          right: 10px;
          z-index: 3;
          background-color: #eff6ff;
          border-color: #3b82f6;
        }
        
        .document.tailored:before, .document.tailored:after {
          background-color: #3b82f6;
        }
        
        .document-shadow {
          position: absolute;
          width: 60px;
          height: 80px;
          background-color: rgba(148, 163, 184, 0.2);
          border-radius: 4px;
          top: 50%;
          left: 50%;
          transform: translate(-50%, -50%);
          z-index: 0;
        }
        
        .processing-rays {
          position: absolute;
          top: 50%;
          left: 50%;
          transform: translate(-50%, -50%);
          z-index: 2;
          width: 60px;
          height: 60px;
        }
        
        .ray {
          position: absolute;
          width: 40px;
          height: 2px;
          background: linear-gradient(90deg, rgba(59, 130, 246, 0), rgba(59, 130, 246, 1));
          top: 50%;
          left: 50%;
          animation: ray-animation 2s infinite;
        }
        
        .ray-1 { transform: translate(-50%, -50%) rotate(0deg); animation-delay: 0s; }
        .ray-2 { transform: translate(-50%, -50%) rotate(72deg); animation-delay: 0.1s; }
        .ray-3 { transform: translate(-50%, -50%) rotate(144deg); animation-delay: 0.2s; }
        .ray-4 { transform: translate(-50%, -50%) rotate(216deg); animation-delay: 0.3s; }
        .ray-5 { transform: translate(-50%, -50%) rotate(288deg); animation-delay: 0.4s; }
        
        .processing-title {
          text-align: center;
          font-size: 1.5rem;
          font-weight: 600;
          color: #1e293b;
          margin-bottom: 1.5rem;
        }
        
        .stage-indicator {
          display: flex;
          align-items: center;
          gap: 1rem;
          margin-bottom: 1.5rem;
          background-color: #f1f5f9;
          padding: 1rem;
          border-radius: 0.5rem;
          animation: pulse 2s infinite;
        }
        
        .stage-icon {
          font-size: 1.75rem;
        }
        
        .processing-message {
          margin: 0;
          font-size: 1.05rem;
          color: #334155;
          flex: 1;
        }
        
        .progress-container {
          margin-bottom: 1.5rem;
        }
        
        .progress-bar {
          width: 100%;
          height: 12px;
          background-color: #e2e8f0;
          border-radius: 6px;
          overflow: hidden;
          margin-bottom: 0.5rem;
        }
        
        .progress-fill {
          height: 100%;
          background: linear-gradient(90deg, #3b82f6, #2563eb);
          border-radius: 6px;
          transition: width 0.3s ease;
        }
        
        .progress-fill.complete {
          background: linear-gradient(90deg, #10b981, #059669);
        }
        
        .progress-label {
          text-align: right;
          font-size: 0.9rem;
          color: #64748b;
        }
        
        .processing-footer {
          display: flex;
          justify-content: center;
          align-items: center;
          font-size: 0.9rem;
          color: #64748b;
          gap: 0.5rem;
        }
        
        .ai-model-name {
          font-weight: 600;
          color: #3b82f6;
        }
        
        .error-details {
          margin-top: 1.5rem;
          padding: 1rem;
          background-color: #fff5f5;
          border-left: 4px solid #ef4444;
          border-radius: 0.25rem;
        }
        
        .error-message {
          margin: 0;
          color: #b91c1c;
          font-size: 0.95rem;
        }
        
        .completion-message {
          margin-top: 1.5rem;
          text-align: center;
          font-size: 1.1rem;
          color: #10b981;
          font-weight: 500;
          animation: fade-in 0.5s ease;
        }
        
        .processing-overlay.error .progress-fill {
          background: linear-gradient(90deg, #ef4444, #b91c1c);
        }
        
        .processing-overlay.error .processing-title {
          color: #b91c1c;
        }
        
        .processing-overlay.complete .stage-indicator {
          background-color: #ecfdf5;
          animation: none;
        }
        
        .processing-overlay.error .stage-indicator {
          background-color: #fff5f5;
          animation: none;
        }
        
        @keyframes ray-animation {
          0% { transform: translate(-50%, -50%) rotate(0deg) scale(0.8); opacity: 0.3; }
          50% { transform: translate(-50%, -50%) rotate(180deg) scale(1.2); opacity: 0.8; }
          100% { transform: translate(-50%, -50%) rotate(360deg) scale(0.8); opacity: 0.3; }
        }
        
        @keyframes fade-in {
          from { opacity: 0; }
          to { opacity: 1; }
        }
        
        @keyframes slide-up {
          from { transform: translateY(30px); opacity: 0; }
          to { transform: translateY(0); opacity: 1; }
        }
        
        @keyframes pulse {
          0%, 100% { background-color: #f1f5f9; }
          50% { background-color: #f8fafc; }
        }
        
        @media (max-width: 768px) {
          .processing-modal {
            padding: 1.5rem;
          }
          
          .processing-icon {
            width: 100px;
            height: 100px;
          }
          
          .processing-title {
            font-size: 1.3rem;
          }
          
          .stage-indicator {
            padding: 0.75rem;
          }
          
          .stage-icon {
            font-size: 1.5rem;
          }
          
          .processing-message {
            font-size: 0.95rem;
          }
        }
      `}</style>
    </div>
  );
};

export default ProcessingOverlay;