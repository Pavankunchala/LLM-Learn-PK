import React, { useEffect, useState } from 'react';

const ModelSelection = ({ selectedModel, onChange, onBack, onProcess, isProcessing }) => {
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    // Fetch available models from the backend
    const fetchModels = async () => {
      try {
        setLoading(true);
        const response = await fetch('http://localhost:5000/models');
        
        if (!response.ok) {
          throw new Error('Failed to fetch models');
        }
        
        const data = await response.json();
        
        if (data.models && Array.isArray(data.models)) {
          setModels(data.models);
          
          // If no model is selected and we have models, select the first one
          if (!selectedModel && data.models.length > 0) {
            onChange(data.models[0].name);
          }
          
          // If the selected model isn't in the list, select the first available
          if (selectedModel && !data.models.find(m => m.name === selectedModel) && data.models.length > 0) {
            onChange(data.models[0].name);
          }
        } else {
          throw new Error('Invalid model data received');
        }
      } catch (err) {
        console.error('Error fetching models:', err);
        setError(err.message);
        
        // Fallback models in case the API fails
        setModels([
          { 
            name: 'llama3', 
            description: 'Meta\'s Llama 3 - Good general-purpose model (Fallback)',
            size: 'Unknown'
          },
          { 
            name: 'mistral', 
            description: 'Mistral AI\'s base model - Excellent general-purpose model (Fallback)',
            size: 'Unknown'
          }
        ]);
      } finally {
        setLoading(false);
      }
    };

    fetchModels();
  }, [selectedModel, onChange]);

  // Format file size to human readable format
  const formatFileSize = (bytes) => {
    if (!bytes || isNaN(bytes)) return 'Unknown size';
    
    const units = ['B', 'KB', 'MB', 'GB', 'TB'];
    let size = bytes;
    let unitIndex = 0;
    
    while (size >= 1024 && unitIndex < units.length - 1) {
      size /= 1024;
      unitIndex++;
    }
    
    return `${size.toFixed(1)} ${units[unitIndex]}`;
  };

  // Get model icon based on name
  const getModelIcon = (modelName) => {
    const name = modelName.toLowerCase();
    if (name.includes('llama')) return '🦙';
    if (name.includes('mistral')) return '🌪️';
    if (name.includes('phi')) return 'Φ';
    if (name.includes('gemma')) return '💎';
    if (name.includes('mixtral')) return '🔄';
    if (name.includes('dolphin')) return '🐬';
    if (name.includes('orca')) return '🐋';
    if (name.includes('wizard')) return '🧙';
    if (name.includes('neural')) return '🧠';
    if (name.includes('code')) return '👨‍💻';
    return '🤖';
  };

  // Get model card style based on name (for gradient background)
  const getModelCardStyle = (modelName) => {
    const name = modelName.toLowerCase();
    if (name.includes('llama')) 
      return { background: 'linear-gradient(135deg, #f6d365 0%, #fda085 100%)' };
    if (name.includes('mistral')) 
      return { background: 'linear-gradient(135deg, #89f7fe 0%, #66a6ff 100%)' };
    if (name.includes('phi')) 
      return { background: 'linear-gradient(135deg, #a1c4fd 0%, #c2e9fb 100%)' };
    if (name.includes('gemma')) 
      return { background: 'linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%)' };
    if (name.includes('mixtral')) 
      return { background: 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)' };
    if (name.includes('orca')) 
      return { background: 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)' };
    
    return { background: 'linear-gradient(135deg, #e0c3fc 0%, #8ec5fc 100%)' };
  };

  return (
    <div className="model-selection">
      <div className="section-header">
        <h2 className="section-title">Select AI Model</h2>
        <p className="section-description">
          Choose the AI model that will tailor your resume. Different models have different strengths.
        </p>
      </div>
      
      {loading ? (
        <div className="loading-container">
          <div className="spinner"></div>
          <p className="loading-text">Discovering available models...</p>
        </div>
      ) : error ? (
        <div className="error-container">
          <div className="error-icon">⚠️</div>
          <div className="error-content">
            <h3 className="error-title">Error loading models</h3>
            <p className="error-message">{error}</p>
            <p className="error-help">Using fallback models instead. Some local models may not appear.</p>
          </div>
        </div>
      ) : (
        <>
          {models.length === 0 ? (
            <div className="no-models-message">
              <div className="no-models-icon">📦</div>
              <h3 className="no-models-title">No models found</h3>
              <p>Please make sure Ollama is running and has models installed.</p>
              <div className="code-block">
                <code>ollama pull modelname</code>
                <button 
                  className="copy-button"
                  onClick={() => navigator.clipboard.writeText('ollama pull llama3')}
                  title="Copy to clipboard"
                >
                  📋
                </button>
              </div>
            </div>
          ) : (
            <div className="models-grid">
              {models.map((model) => (
                <div 
                  key={model.name}
                  className={`model-card ${selectedModel === model.name ? 'selected' : ''}`}
                  onClick={() => onChange(model.name)}
                >
                  <div className="model-header" style={getModelCardStyle(model.name)}>
                    <div className="model-icon">{getModelIcon(model.name)}</div>
                    <div className="model-name">{model.name}</div>
                    {selectedModel === model.name && 
                      <div className="selection-badge">Selected</div>
                    }
                  </div>
                  <div className="model-body">
                    <p className="model-description">{model.description || 'No description available'}</p>
                    <div className="model-stats">
                      <div className="stat-item">
                        <div className="stat-icon">💾</div>
                        <div className="stat-value">{formatFileSize(model.size)}</div>
                      </div>
                      {model.modified_at && (
                        <div className="stat-item">
                          <div className="stat-icon">🔄</div>
                          <div className="stat-value">
                            {new Date(model.modified_at).toLocaleDateString()}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                  <div className="model-footer">
                    <div className="model-capability-badge">
                      <span>Performance: </span>
                      {model.name.includes('70b') ? 'Excellent' : 
                       model.name.includes('13b') ? 'Very Good' : 'Good'}
                    </div>
                    <div className="model-capability-badge">
                      <span>Speed: </span>
                      {model.name.includes('8b') ? 'Fast' : 
                       model.name.includes('7b') ? 'Fast' :
                       model.name.includes('3b') ? 'Very Fast' : 'Moderate'}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </>
      )}
      
      <div className="action-buttons">
        <button 
          className="btn btn-secondary" 
          onClick={onBack}
          disabled={isProcessing}
        >
          <span className="btn-icon">←</span>
          Back
        </button>
        
        <button 
          className="btn btn-primary" 
          onClick={onProcess}
          disabled={isProcessing || !selectedModel}
        >
          {isProcessing ? (
            <>
              <span className="spinner-small"></span>
              Processing...
            </>
          ) : (
            <>
              <span className="btn-icon">✨</span>
              Tailor Resume
            </>
          )}
        </button>
      </div>

      <style jsx>{`
        .model-selection {
          max-width: 900px;
          margin: 0 auto;
        }
        
        .section-header {
          margin-bottom: 2rem;
          text-align: center;
        }
        
        .section-title {
          font-size: 1.8rem;
          margin-bottom: 0.5rem;
          color: var(--gray-900);
          position: relative;
          display: inline-block;
        }
        
        .section-title:after {
          content: '';
          position: absolute;
          bottom: -8px;
          left: 50%;
          transform: translateX(-50%);
          width: 60px;
          height: 3px;
          background: linear-gradient(90deg, var(--primary), var(--accent, purple));
          border-radius: 3px;
        }
        
        .section-description {
          color: var(--gray-600);
          font-size: 1rem;
          max-width: 600px;
          margin: 1rem auto 0;
        }
        
        /* Loading state */
        .loading-container {
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          padding: 3rem;
          background-color: var(--gray-50);
          border-radius: 0.75rem;
          border: 1px dashed var(--gray-300);
        }
        
        .spinner {
          width: 40px;
          height: 40px;
          border: 3px solid rgba(59, 130, 246, 0.2);
          border-radius: 50%;
          border-top-color: var(--primary);
          animation: spin 1s linear infinite;
          margin-bottom: 1rem;
        }
        
        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
        
        .loading-text {
          color: var(--gray-600);
          font-size: 1rem;
        }
        
        /* Error state */
        .error-container {
          display: flex;
          gap: 1.5rem;
          padding: 1.5rem;
          background-color: #fff5f5;
          border-left: 4px solid #f56565;
          border-radius: 0.5rem;
          margin-bottom: 2rem;
        }
        
        .error-icon {
          font-size: 1.5rem;
        }
        
        .error-title {
          color: #c53030;
          margin-bottom: 0.5rem;
          font-size: 1.1rem;
        }
        
        .error-message {
          color: #c53030;
          margin-bottom: 0.5rem;
        }
        
        .error-help {
          color: #718096;
          font-size: 0.9rem;
        }
        
        /* No models message */
        .no-models-message {
          text-align: center;
          padding: 3rem;
          background-color: var(--gray-50);
          border-radius: 0.75rem;
          border: 1px dashed var(--gray-300);
        }
        
        .no-models-icon {
          font-size: 2.5rem;
          margin-bottom: 1rem;
        }
        
        .no-models-title {
          margin-bottom: 1rem;
          color: var(--gray-700);
        }
        
        .code-block {
          display: flex;
          align-items: center;
          justify-content: center;
          background-color: var(--gray-800);
          color: white;
          padding: 0.75rem 1rem;
          border-radius: 0.5rem;
          margin: 1rem auto;
          max-width: 300px;
          position: relative;
        }
        
        .copy-button {
          background: none;
          border: none;
          color: white;
          cursor: pointer;
          position: absolute;
          right: 0.5rem;
          opacity: 0.6;
          transition: opacity 0.2s;
        }
        
        .copy-button:hover {
          opacity: 1;
        }
        
        /* Models grid */
        .models-grid {
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
          gap: 1.5rem;
          margin-bottom: 2rem;
        }
        
        .model-card {
          background-color: white;
          border-radius: 1rem;
          overflow: hidden;
          box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
          cursor: pointer;
          transition: all 0.3s ease;
          height: 100%;
          display: flex;
          flex-direction: column;
          border: 1px solid var(--gray-200);
        }
        
        .model-card:hover {
          transform: translateY(-5px);
          box-shadow: 0 12px 20px rgba(0, 0, 0, 0.1);
        }
        
        .model-card.selected {
          box-shadow: 0 10px 25px rgba(59, 130, 246, 0.4);
          transform: translateY(-5px) scale(1.02);
          border: 2px solid var(--primary);
        }
        
        .model-header {
          padding: 1.5rem;
          position: relative;
          display: flex;
          align-items: center;
          gap: 1rem;
          color: white;
          text-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
        }
        
        .model-icon {
          font-size: 2rem;
          background-color: rgba(255, 255, 255, 0.25);
          width: 50px;
          height: 50px;
          display: flex;
          align-items: center;
          justify-content: center;
          border-radius: 12px;
          backdrop-filter: blur(5px);
          box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
        }
        
        .model-name {
          font-size: 1.25rem;
          font-weight: 600;
          flex: 1;
          text-transform: capitalize;
        }
        
        .selection-badge {
          position: absolute;
          top: 10px;
          right: 10px;
          background-color: white;
          color: var(--primary);
          font-size: 0.75rem;
          font-weight: 600;
          padding: 0.25rem 0.5rem;
          border-radius: 10px;
          box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
        }
        
        .model-body {
          padding: 1.5rem;
          flex: 1;
          display: flex;
          flex-direction: column;
        }
        
        .model-description {
          color: var(--gray-600);
          font-size: 0.95rem;
          line-height: 1.5;
          margin-bottom: 1.5rem;
          flex: 1;
        }
        
        .model-stats {
          display: flex;
          gap: 1rem;
          justify-content: space-between;
        }
        
        .stat-item {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          font-size: 0.85rem;
          color: var(--gray-600);
        }
        
        .stat-icon {
          font-size: 1rem;
        }
        
        .model-footer {
          padding: 1rem 1.5rem;
          border-top: 1px solid var(--gray-100);
          display: flex;
          flex-wrap: wrap;
          gap: 0.5rem;
        }
        
        .model-capability-badge {
          background-color: var(--gray-100);
          color: var(--gray-700);
          font-size: 0.8rem;
          padding: 0.25rem 0.5rem;
          border-radius: 0.25rem;
        }
        
        .model-capability-badge span {
          color: var(--gray-500);
        }
        
        /* Action buttons */
        .action-buttons {
          display: flex;
          justify-content: space-between;
          margin-top: 2rem;
        }
        
        .btn {
          display: inline-flex;
          align-items: center;
          gap: 0.5rem;
          font-weight: 500;
          padding: 0.75rem 1.75rem;
          border-radius: 0.5rem;
          cursor: pointer;
          transition: all 0.3s ease;
          border: none;
          font-size: 1rem;
        }
        
        .btn-icon {
          font-size: 1.1rem;
        }
        
        .btn-primary {
          background: linear-gradient(135deg, var(--primary), var(--primary-dark, #1e40af));
          color: white;
          box-shadow: 0 4px 10px rgba(59, 130, 246, 0.3);
        }
        
        .btn-primary:hover:not(:disabled) {
          transform: translateY(-2px);
          box-shadow: 0 8px 15px rgba(59, 130, 246, 0.4);
          filter: brightness(1.1);
        }
        
        .btn-secondary {
          background-color: var(--gray-100);
          color: var(--gray-700);
        }
        
        .btn-secondary:hover:not(:disabled) {
          background-color: var(--gray-200);
          transform: translateY(-2px);
          box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        .btn:disabled {
          opacity: 0.6;
          cursor: not-allowed;
          transform: none !important;
          box-shadow: none !important;
        }
        
        .spinner-small {
          width: 1rem;
          height: 1rem;
          border: 2px solid rgba(255, 255, 255, 0.3);
          border-radius: 50%;
          border-top-color: white;
          animation: spin 1s linear infinite;
          margin-right: 0.5rem;
        }
        
        /* Responsive styles */
        @media (max-width: 768px) {
          .models-grid {
            grid-template-columns: 1fr;
          }
          
          .model-card {
            max-width: 100%;
          }
        }
      `}</style>
    </div>
  );
};

export default ModelSelection;