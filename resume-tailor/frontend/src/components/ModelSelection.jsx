import React, { useEffect, useState } from 'react';

// Optional debugging component - uncomment to use
// import ModelDebugger from './ModelDebugger';

const ModelSelection = ({ selectedModel, onChange, onBack, onProcess, isProcessing }) => {
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [fetchRetries, setFetchRetries] = useState(0);
  const [sortBy, setSortBy] = useState('name'); // New state for sorting
  const [sortOrder, setSortOrder] = useState('asc'); // New state for sort order

  useEffect(() => {
    fetchModels();
  }, [selectedModel, onChange, fetchRetries]);

  const fetchModels = async () => {
    try {
      setLoading(true);
      const response = await fetch('http://localhost:5000/models');
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      console.log("Models API response:", data); // Debug output
      
      // Handle both response formats (direct models array or wrapped response)
      const modelsData = data.models || data;
      
      if (modelsData && Array.isArray(modelsData)) {
        // Filter out any invalid models (without name property)
        const validModels = modelsData.filter(model => model && model.name);
        
        if (validModels.length === 0) {
          throw new Error('No valid models returned from the server');
        }
        
        // Add descriptions if missing and enhance with additional properties
        const enhancedModels = validModels.map(model => ({
          ...model,
          description: model.description || getDefaultDescription(model.name),
          performance: estimatePerformance(model.name),
          speed: getModelSpeed(model.name),
          family: detectModelFamily(model.name),
          tags: generateModelTags(model)
        }));
        
        console.log("Enhanced models:", enhancedModels); // Debug output
        setModels(enhancedModels);
        
        // If no model is selected and we have models, select the first one
        if ((!selectedModel || selectedModel === '') && enhancedModels.length > 0) {
          console.log("Setting initial model:", enhancedModels[0].name);
          onChange(enhancedModels[0].name);
        }
      } else {
        throw new Error('Invalid model data received');
      }
    } catch (err) {
      console.error('Error fetching models:', err);
      setError(err.message);
      
      // Retry logic (only retry a few times)
      if (fetchRetries < 2) {
        setFetchRetries(prev => prev + 1);
        setTimeout(fetchModels, 1000); // Try again after 1 second
        return;
      }
      
      // Fallback models if all retries fail
      const fallbackModels = [
        { 
          name: 'llama3', 
          description: 'Meta\'s Llama 3 - Good general-purpose model (Fallback)',
          performance: 'Good',
          speed: 'Fast',
          family: 'llama',
          tags: ['fallback', 'general-purpose']
        },
        { 
          name: 'mistral', 
          description: 'Mistral AI\'s base model - Excellent general-purpose model (Fallback)',
          performance: 'Very Good',
          speed: 'Moderate',
          family: 'mistral',
          tags: ['fallback', 'general-purpose']
        }
      ];
      
      setModels(fallbackModels);
      
      // Select the first fallback model only if no model is already selected
      if (!selectedModel || selectedModel === '') {
        console.log("Setting fallback model:", fallbackModels[0].name);
        onChange('llama3');
      }
    } finally {
      setLoading(false);
    }
  };

  // Function to detect model family from name
  const detectModelFamily = (modelName) => {
    if (!modelName) return 'unknown';
    
    const name = modelName.toLowerCase();
    if (name.includes('llama')) return 'llama';
    if (name.includes('mistral')) return 'mistral';
    if (name.includes('phi')) return 'phi';
    if (name.includes('gemma')) return 'gemma';
    if (name.includes('qwen')) return 'qwen';
    if (name.includes('mixtral')) return 'mixtral';
    if (name.includes('deepseek')) return 'deepseek';
    if (name.includes('orca')) return 'orca';
    
    return 'other';
  };

  // Generate tags for model
  const generateModelTags = (model) => {
    const tags = [];
    const name = model.name.toLowerCase();
    
    // Add size-based tags
    if (name.includes('3b') || name.includes('7b') || model.parameter_size?.includes('3') || model.parameter_size?.includes('7')) {
      tags.push('small');
    } else if (name.includes('13b') || name.includes('14b') || model.parameter_size?.includes('13') || model.parameter_size?.includes('14')) {
      tags.push('medium');
    } else if (name.includes('70b') || model.parameter_size?.includes('70')) {
      tags.push('large');
    }
    
    // Add capability tags
    if (name.includes('vision')) tags.push('vision');
    if (name.includes('coder')) tags.push('coding');
    if (name.includes('chat')) tags.push('chat');
    if (name.includes('instruct')) tags.push('instruction');
    
    // Add quantization tag if available
    if (model.quantization) {
      const quant = model.quantization.toLowerCase();
      if (quant.includes('q4')) tags.push('4-bit');
      else if (quant.includes('q5')) tags.push('5-bit');
      else if (quant.includes('q8')) tags.push('8-bit');
      else if (quant.includes('f16')) tags.push('16-bit');
    }
    
    return tags;
  };

  // Get default description for a model
  const getDefaultDescription = (modelName) => {
    if (!modelName) return 'No description available';
    
    const name = modelName.toLowerCase();
    if (name.includes('llama')) return 'Meta\'s Llama model family - versatile general-purpose model';
    if (name.includes('mistral')) return 'Mistral AI\'s model - excellent reasoning and instruction following';
    if (name.includes('phi')) return 'Microsoft Phi model - compact and efficient';
    if (name.includes('gemma')) return 'Google\'s Gemma model - lightweight and performant';
    if (name.includes('qwen')) return 'Qwen model by Alibaba - strong multilingual capabilities';
    if (name.includes('mixtral')) return 'Mistral\'s mixture of experts model - high capability';
    if (name.includes('deepseek')) return 'DeepSeek model - state-of-the-art capabilities';
    
    return 'General-purpose language model';
  };
  
  // Estimate performance based on model name
  const estimatePerformance = (modelName) => {
    if (!modelName) return 'Unknown';
    
    const name = modelName.toLowerCase();
    // Check for model size indicators in name
    if (name.includes('70b') || name.includes('opus')) return 'Excellent';
    if (name.includes('13b') || name.includes('14b') || name.includes('medium')) return 'Very Good';
    if (name.includes('8b') || name.includes('7b') || name.includes('small')) return 'Good';
    if (name.includes('3b') || name.includes('tiny')) return 'Basic';
    
    // Default performance estimation
    return 'Good';
  };

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
    if (!modelName) return '🤖'; // Default icon
    
    const name = modelName.toLowerCase();
    if (name.includes('llama')) return '🦙';
    if (name.includes('mistral')) return '🌪️';
    if (name.includes('phi')) return 'Φ';
    if (name.includes('gemma')) return '💎';
    if (name.includes('qwen')) return '🔍';
    if (name.includes('mixtral')) return '🔄';
    if (name.includes('dolphin')) return '🐬';
    if (name.includes('orca')) return '🐋';
    if (name.includes('wizard')) return '🧙';
    if (name.includes('neural')) return '🧠';
    if (name.includes('code')) return '👨‍💻';
    if (name.includes('deepseek')) return '🔎';
    
    return '🤖';
  };

  // Get model card style based on name (for gradient background)
  const getModelCardStyle = (modelName) => {
    if (!modelName) {
      return { background: 'linear-gradient(135deg, #e0c3fc 0%, #8ec5fc 100%)' }; // Default gradient
    }
    
    const name = modelName.toLowerCase();
    if (name.includes('llama')) 
      return { background: 'linear-gradient(135deg, #f6d365 0%, #fda085 100%)' };
    if (name.includes('mistral')) 
      return { background: 'linear-gradient(135deg, #89f7fe 0%, #66a6ff 100%)' };
    if (name.includes('phi')) 
      return { background: 'linear-gradient(135deg, #a1c4fd 0%, #c2e9fb 100%)' };
    if (name.includes('gemma')) 
      return { background: 'linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%)' };
    if (name.includes('qwen')) 
      return { background: 'linear-gradient(135deg, #fbc2eb 0%, #a6c1ee 100%)' };
    if (name.includes('mixtral')) 
      return { background: 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)' };
    if (name.includes('orca')) 
      return { background: 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)' };
    if (name.includes('deepseek'))
      return { background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)' };
    
    return { background: 'linear-gradient(135deg, #e0c3fc 0%, #8ec5fc 100%)' }; // Default gradient
  };

  // Get model speed rating
  const getModelSpeed = (modelName) => {
    if (!modelName) return 'Unknown';
    
    const name = modelName.toLowerCase();
    if (name.includes('3b') || name.includes('tiny') || name.includes('small'))
      return 'Very Fast';
    if (name.includes('7b') || name.includes('8b'))
      return 'Fast';
    if (name.includes('13b') || name.includes('14b') || name.includes('medium'))
      return 'Moderate';
    if (name.includes('70b') || name.includes('large') || name.includes('opus'))
      return 'Slower';
    
    return 'Moderate';
  };

  // Enhanced model selection handler
  const handleModelSelection = (modelName) => {
    console.log("Model selected:", modelName); // Debug output
    onChange(modelName);
  };

  // Sort models function
  const sortModels = (a, b) => {
    if (sortBy === 'name') {
      return sortOrder === 'asc' 
        ? a.name.localeCompare(b.name)
        : b.name.localeCompare(a.name);
    } else if (sortBy === 'size') {
      const sizeA = a.size || 0;
      const sizeB = b.size || 0;
      return sortOrder === 'asc' ? sizeA - sizeB : sizeB - sizeA;
    } else if (sortBy === 'performance') {
      const perfMap = {
        'Excellent': 4,
        'Very Good': 3,
        'Good': 2,
        'Basic': 1,
        'Unknown': 0
      };
      const perfA = perfMap[a.performance] || 0;
      const perfB = perfMap[b.performance] || 0;
      return sortOrder === 'asc' ? perfA - perfB : perfB - perfA;
    }
    return 0;
  };

  // Handle sorting change
  const handleSortChange = (newSortBy) => {
    if (sortBy === newSortBy) {
      // Toggle order if clicking the same sort option
      setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc');
    } else {
      // Set new sort option with default ascending order
      setSortBy(newSortBy);
      setSortOrder('asc');
    }
  };

  // Safely render a model card
  const renderModelCard = (model) => {
    if (!model || !model.name) return null;
    
    return (
      <div 
        key={model.name}
        className={`model-card ${selectedModel === model.name ? 'selected' : ''}`}
        onClick={() => handleModelSelection(model.name)}
      >
        <div className="model-header" style={getModelCardStyle(model.name)}>
          <div className="model-icon">{getModelIcon(model.name)}</div>
          <div className="model-name">{model.name}</div>
          {selectedModel === model.name && 
            <div className="selection-badge">Selected</div>
          }
        </div>
        <div className="model-body">
          <p className="model-description">{model.description || getDefaultDescription(model.name)}</p>
          <div className="model-stats">
            <div className="stat-item">
              <div className="stat-icon">💾</div>
              <div className="stat-value">{formatFileSize(model.size)}</div>
            </div>
            {model.parameter_size && (
              <div className="stat-item" title="Model parameter size">
                <div className="stat-icon">🧠</div>
                <div className="stat-value">{model.parameter_size}</div>
              </div>
            )}
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
        <div className="model-tags">
          {model.tags && model.tags.map((tag, idx) => (
            <span key={idx} className="model-tag">{tag}</span>
          ))}
        </div>
        <div className="model-footer">
          <div className="model-capability-badge">
            <span>Performance: </span>
            {model.performance || estimatePerformance(model.name)}
          </div>
          <div className="model-capability-badge">
            <span>Speed: </span>
            {model.speed || getModelSpeed(model.name)}
          </div>
        </div>
      </div>
    );
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
                <code>ollama pull llama3</code>
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
            <>
              <div className="models-tools">
                <div className="model-count">
                  <span className="model-count-badge">{models.length}</span> models available
                </div>
                <div className="sort-options">
                  <span>Sort by: </span>
                  <button 
                    className={`sort-button ${sortBy === 'name' ? 'active' : ''}`}
                    onClick={() => handleSortChange('name')}
                  >
                    Name {sortBy === 'name' && (sortOrder === 'asc' ? '↑' : '↓')}
                  </button>
                  <button 
                    className={`sort-button ${sortBy === 'size' ? 'active' : ''}`}
                    onClick={() => handleSortChange('size')}
                  >
                    Size {sortBy === 'size' && (sortOrder === 'asc' ? '↑' : '↓')}
                  </button>
                  <button 
                    className={`sort-button ${sortBy === 'performance' ? 'active' : ''}`}
                    onClick={() => handleSortChange('performance')}
                  >
                    Performance {sortBy === 'performance' && (sortOrder === 'asc' ? '↑' : '↓')}
                  </button>
                </div>
              </div>
              <div className="models-grid">
                {/* Sort models based on current sort criteria */}
                {[...models].sort(sortModels).map(model => 
                  model && model.name ? renderModelCard(model) : null
                )}
              </div>
            </>
          )}
        </>
      )}
      
      {/* Uncomment to enable the model debugger */}
      {/* <ModelDebugger selectedModel={selectedModel} availableModels={models} /> */}
      
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
        
        /* Models tools */
        .models-tools {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 1.5rem;
          padding: 0 0.5rem;
        }
        
        .model-count {
          font-size: 0.9rem;
          color: var(--gray-600);
          display: flex;
          align-items: center;
          gap: 0.5rem;
        }
        
        .model-count-badge {
          display: inline-flex;
          align-items: center;
          justify-content: center;
          background-color: var(--primary);
          color: white;
          width: 24px;
          height: 24px;
          border-radius: 12px;
          font-weight: 600;
          font-size: 0.8rem;
        }
        
        .sort-options {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          font-size: 0.9rem;
          color: var(--gray-600);
        }
        
        .sort-button {
          background: none;
          border: none;
          padding: 0.25rem 0.5rem;
          cursor: pointer;
          font-size: 0.85rem;
          border-radius: 4px;
          color: var(--gray-600);
          transition: all 0.2s;
        }
        
        .sort-button:hover {
          background-color: var(--gray-100);
          color: var(--gray-900);
        }
        
        .sort-button.active {
          background-color: var(--gray-200);
          color: var(--gray-900);
          font-weight: 500;
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
          position: relative;
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
          word-break: break-word;
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
          flex-wrap: wrap;
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
        
        /* Model tags */
        .model-tags {
          padding: 0 1.5rem;
          display: flex;
          flex-wrap: wrap;
          gap: 0.5rem;
          margin-bottom: 1rem;
        }
        
        .model-tag {
          display: inline-block;
          font-size: 0.75rem;
          background-color: var(--gray-100);
          color: var(--gray-700);
          padding: 0.15rem 0.5rem;
          border-radius: 20px;
          border: 1px solid var(--gray-200);
        }
        
        .model-footer {
          padding: 1rem 1.5rem;
          border-top: 1px solid var(--gray-100);
          display: flex;
          flex-wrap: wrap;
          gap: 0.5rem;
          background-color: var(--gray-50);
        }
        
        .model-capability-badge {
          background-color: white;
          color: var(--gray-700);
          font-size: 0.8rem;
          padding: 0.25rem 0.5rem;
          border-radius: 0.25rem;
          border: 1px solid var(--gray-200);
          box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
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
          
          .models-tools {
            flex-direction: column;
            gap: 1rem;
            align-items: flex-start;
          }
          
          .sort-options {
            flex-wrap: wrap;
          }
        }
      `}</style>
    </div>
  );
};

export default ModelSelection;