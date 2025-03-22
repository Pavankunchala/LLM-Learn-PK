import React, { useState, useEffect, useRef } from 'react';
import PDFViewer from './PDFViewer';
import { downloadPdfFromBase64, downloadTextContent, analyzePdfData } from '../utils/pdfDownloadUtils';

const ResultPreview = ({ result, onBack, onReset }) => {
  const [activeTab, setActiveTab] = useState('pdf');
  const [pdfDebugInfo, setPdfDebugInfo] = useState(null);
  const [pdfDataUrl, setPdfDataUrl] = useState(null);
  const [showDebug, setShowDebug] = useState(false);
  const [downloadAnimation, setDownloadAnimation] = useState('');
  const [pdfZoom, setPdfZoom] = useState(1.0);
  const [copied, setCopied] = useState(false);
  const [summaryExpanded, setSummaryExpanded] = useState(false);
  
  const pdfContainerRef = useRef(null);
  const latexCodeRef = useRef(null);
  
  // Process the PDF data when the component loads
  useEffect(() => {
    console.log("Processing PDF data...");
    
    if (result?.pdf) {
      try {
        // Analyze the PDF data
        const analysis = analyzePdfData(result.pdf);
        setPdfDebugInfo(analysis);
        
        // Create a data URL if the PDF data is valid base64
        if (analysis.valid) {
          // Check if it already has the data URL prefix
          const dataUrl = result.pdf.startsWith('data:application/pdf;base64,')
            ? result.pdf
            : `data:application/pdf;base64,${result.pdf}`;
          
          setPdfDataUrl(dataUrl);
        } else {
          console.error("Invalid PDF data:", analysis);
        }
      } catch (error) {
        console.error("Error processing PDF data:", error);
        setPdfDebugInfo({ error: error.message });
      }
    } else {
      console.warn("No PDF data available in result");
      setPdfDebugInfo({ error: "No PDF data available" });
    }
  }, [result]);

  const handleDownloadLatex = () => {
    if (result?.latex) {
      setDownloadAnimation('latex');
      
      const success = downloadTextContent(
        result.latex,
        'tailored_resume.tex',
        'application/x-tex'
      );
      
      setTimeout(() => setDownloadAnimation(''), 1500);
      
      if (!success) {
        alert('Failed to download LaTeX. Please try again or copy the code manually.');
      }
    } else {
      alert('No LaTeX content available to download.');
    }
  };

  const handleDownloadPdf = () => {
    if (!result?.pdf) {
      alert('PDF is not available. Try downloading the LaTeX code and compiling it manually.');
      return;
    }
    
    setDownloadAnimation('pdf');
    
    const success = downloadPdfFromBase64(result.pdf, 'tailored_resume.pdf');
    
    setTimeout(() => setDownloadAnimation(''), 1500);
    
    if (!success) {
      // If direct download fails, try offering the data URL for the user to save manually
      if (pdfDataUrl) {
        const shouldOpenInNewTab = window.confirm(
          'Direct download failed. Would you like to open the PDF in a new tab? ' +
          'You can then use your browser\'s save function to download it.'
        );
        
        if (shouldOpenInNewTab) {
          window.open(pdfDataUrl, '_blank');
        }
      } else {
        alert('Failed to download PDF. Please try downloading the LaTeX code instead.');
      }
    }
  };

  const handleCopyLatex = () => {
    if (result?.latex) {
      navigator.clipboard.writeText(result.latex)
        .then(() => {
          setCopied(true);
          setTimeout(() => setCopied(false), 2000);
        })
        .catch(err => {
          console.error('Error copying text:', err);
          alert('Failed to copy to clipboard. Please try manually selecting the text.');
        });
    }
  };

  const handleZoomIn = () => {
    setPdfZoom(prevZoom => Math.min(prevZoom + 0.25, 2.5));
  };

  const handleZoomOut = () => {
    setPdfZoom(prevZoom => Math.max(prevZoom - 0.25, 0.5));
  };

  const handleResetZoom = () => {
    setPdfZoom(1.0);
  };

  // Force debug tab to be visible if we're encountering PDF issues
  useEffect(() => {
    if (pdfDebugInfo && pdfDebugInfo.error) {
      setShowDebug(true);
    }
  }, [pdfDebugInfo]);

  // Format the summary text with better structure
  const formatSummary = (summary) => {
    if (!summary) return [];
    
    // Split by line breaks and filter out empty lines
    const lines = summary.split('\n').filter(line => line.trim());
    
    // Group lines into sections (assuming blank lines separate sections)
    const sections = [];
    let currentSection = [];
    
    lines.forEach(line => {
      if (line.trim()) {
        currentSection.push(line);
      } else if (currentSection.length > 0) {
        sections.push([...currentSection]);
        currentSection = [];
      }
    });
    
    // Add the last section if not empty
    if (currentSection.length > 0) {
      sections.push(currentSection);
    }
    
    return sections;
  };

  const formattedSummary = formatSummary(result?.summary);
  const summaryToShow = summaryExpanded ? formattedSummary : formattedSummary.slice(0, 1);

  return (
    <div className="result-preview">
      <div className="success-banner">
        <div className="success-icon">✅</div>
        <div className="success-message">
          <h3>Resume Successfully Tailored!</h3>
          <p>Your resume has been customized to match the job description</p>
        </div>
      </div>
      
      {/* Summary Card */}
      <div className="summary-card">
        <div className="summary-header" onClick={() => setSummaryExpanded(!summaryExpanded)}>
          <div className="summary-title">
            <span className="summary-icon">📋</span>
            Tailoring Summary
          </div>
          <button className="expand-button">
            {summaryExpanded ? '▲ Show Less' : '▼ Show More'}
          </button>
        </div>
        
        <div className="summary-content">
          {summaryToShow.map((section, sectionIndex) => (
            <div key={sectionIndex} className="summary-section">
              {section.map((line, lineIndex) => {
                // Check if this looks like a header line
                const isHeader = lineIndex === 0 && (line.includes(':') || /^[A-Z]/.test(line));
                
                return isHeader ? (
                  <h4 key={lineIndex} className="summary-section-title">{line}</h4>
                ) : (
                  <div key={lineIndex} className="summary-item">
                    <span className="bullet-point">•</span>
                    <p>{line}</p>
                  </div>
                );
              })}
            </div>
          ))}
          
          {formattedSummary.length > 1 && !summaryExpanded && (
            <button 
              className="show-more-button"
              onClick={() => setSummaryExpanded(true)}
            >
              Show Full Summary ({formattedSummary.length - 1} more sections)
            </button>
          )}
        </div>
      </div>
      
      {/* Warning Banner (if issues detected) */}
      {result?.format_warning && (
        <div className="warning-banner">
          <div className="warning-icon">⚠️</div>
          <div className="warning-message">
            <h4>{result.format_warning}</h4>
            <p>Some formatting issues were detected and automatically corrected. Please review the final resume carefully.</p>
          </div>
        </div>
      )}
      
      <div className="preview-tabs">
        <button
          className={`tab-button ${activeTab === 'pdf' ? 'active' : ''}`}
          onClick={() => setActiveTab('pdf')}
        >
          <span className="tab-icon">📄</span>
          PDF Preview
        </button>
        <button
          className={`tab-button ${activeTab === 'latex' ? 'active' : ''}`}
          onClick={() => setActiveTab('latex')}
        >
          <span className="tab-icon">🔣</span>
          LaTeX Code
        </button>
        {(showDebug || pdfDebugInfo?.error) && (
          <button
            className={`tab-button debug ${activeTab === 'debug' ? 'active' : ''}`}
            onClick={() => setActiveTab('debug')}
          >
            <span className="tab-icon">🛠️</span>
            Debug Info
          </button>
        )}
      </div>
      
      <div className="preview-content">
        {activeTab === 'pdf' ? (
          <div className="pdf-preview" ref={pdfContainerRef}>
            {pdfDataUrl ? (
              <>
                <div className="pdf-toolbar">
                  <div className="zoom-controls">
                    <button className="zoom-button" onClick={handleZoomOut} title="Zoom Out">
                      <span className="zoom-icon">➖</span>
                    </button>
                    <span className="zoom-level">{Math.round(pdfZoom * 100)}%</span>
                    <button className="zoom-button" onClick={handleZoomIn} title="Zoom In">
                      <span className="zoom-icon">➕</span>
                    </button>
                    <button className="zoom-button reset" onClick={handleResetZoom} title="Reset Zoom">
                      <span className="zoom-icon">↺</span>
                    </button>
                  </div>
                  <div className="pdf-info">
                    <span className="pdf-method">
                      Generated with: <strong>{result?.pdf_method || 'Unknown'}</strong>
                    </span>
                  </div>
                </div>
                
                <div className="pdf-viewer-container" style={{ transform: `scale(${pdfZoom})` }}>
                  <PDFViewer 
                    pdfData={pdfDataUrl} 
                    scale={1.0}
                  />
                </div>
              </>
            ) : (
              <div className="pdf-error">
                <div className="error-icon">⚠️</div>
                <div className="error-content">
                  <h4>PDF Preview Unavailable</h4>
                  <p>The LaTeX code could not be compiled into a PDF.</p>
                  <p>You can still download the LaTeX code and compile it manually using an online service like Overleaf.</p>
                  {pdfDebugInfo && pdfDebugInfo.error && (
                    <div className="error-details">Error: {pdfDebugInfo.error}</div>
                  )}
                </div>
              </div>
            )}
          </div>
        ) : activeTab === 'latex' ? (
          <div className="latex-preview">
            <div className="code-header">
              <span className="code-title">LaTeX Source Code</span>
              <button 
                className={`copy-button ${copied ? 'copied' : ''}`}
                onClick={handleCopyLatex}
              >
                {copied ? '✓ Copied!' : 'Copy Code'}
              </button>
            </div>
            <pre className="latex-code" ref={latexCodeRef}>
              {result?.latex || 'No LaTeX code was generated.'}
            </pre>
          </div>
        ) : activeTab === 'debug' ? (
          <div className="debug-view">
            <h3 className="debug-title">Debug Information</h3>
            
            <div className="debug-section">
              <h4 className="debug-subtitle">PDF Data Analysis</h4>
              <pre className="debug-code">
                {pdfDebugInfo ? JSON.stringify(pdfDebugInfo, null, 2) : 'No debug info available'}
              </pre>
            </div>
            
            <div className="debug-section">
              <h4 className="debug-subtitle">Backend Response</h4>
              <pre className="debug-code">
                {JSON.stringify({
                  hasLatex: !!result?.latex,
                  latexLength: result?.latex?.length || 0,
                  hasPdf: !!result?.pdf,
                  pdfLength: result?.pdf?.length || 0,
                  hasSummary: !!result?.summary,
                  summaryLength: result?.summary?.length || 0,
                  pdfMethod: result?.pdf_method || 'unknown',
                  evaluation: result?.evaluation || 'n/a'
                }, null, 2)}
              </pre>
            </div>
            
            {result?.pdf && (
              <div className="debug-section">
                <h4 className="debug-subtitle">PDF Data Sample (first 100 chars)</h4>
                <pre className="debug-code sample-code">
                  {result.pdf.substring(0, 100)}...
                </pre>
              </div>
            )}

            <div className="debug-section help-section">
              <h4 className="debug-subtitle">How to Fix PDF Issues</h4>
              <div className="debug-instructions">
                <p>To fix PDF generation issues:</p>
                <ol className="fix-steps">
                  <li>Install LaTeX on your system (recommended):
                    <ul>
                      <li>Windows: <a href="https://miktex.org/download" target="_blank" rel="noopener noreferrer">MiKTeX</a></li>
                      <li>macOS: <a href="https://www.tug.org/mactex/" target="_blank" rel="noopener noreferrer">MacTeX</a></li>
                      <li>Linux: <code>sudo apt-get install texlive-latex-base texlive-fonts-recommended</code></li>
                    </ul>
                  </li>
                  <li>Install Python dependencies:
                    <ul>
                      <li><code>pip install weasyprint reportlab</code></li>
                    </ul>
                  </li>
                  <li>Restart your Flask server</li>
                </ol>
              </div>
            </div>
          </div>
        ) : null}
      </div>
      
      <div className="download-options">
        <button 
          className={`download-button latex-download ${downloadAnimation === 'latex' ? 'downloading' : ''}`}
          onClick={handleDownloadLatex}
          disabled={!result?.latex || downloadAnimation !== ''}
        >
          <span className="download-icon">📄</span>
          <span className="download-text">
            {downloadAnimation === 'latex' ? (
              <>
                <span className="spinner"></span>
                Downloading...
              </>
            ) : 'Download LaTeX (.tex)'}
          </span>
        </button>
        <button 
          className={`download-button pdf-download ${downloadAnimation === 'pdf' ? 'downloading' : ''}`}
          onClick={handleDownloadPdf}
          disabled={!result?.pdf || downloadAnimation !== ''}
        >
          <span className="download-icon">📑</span>
          <span className="download-text">
            {downloadAnimation === 'pdf' ? (
              <>
                <span className="spinner"></span>
                Downloading...
              </>
            ) : 'Download PDF'}
          </span>
        </button>
      </div>
      
      <div className="action-buttons">
        <button 
          className="btn btn-secondary" 
          onClick={onBack}
        >
          <span className="button-icon back-icon">←</span>
          Back to Model Selection
        </button>
        <button 
          className="btn btn-primary" 
          onClick={onReset}
        >
          <span className="button-icon reset-icon">↻</span>
          Start Over
        </button>
      </div>
      
      {/* Updated styles */}
      <style jsx>{`
        .result-preview {
          max-width: 900px;
          margin: 0 auto;
        }
        
        .success-banner {
          display: flex;
          align-items: center;
          gap: 1.5rem;
          background: linear-gradient(135deg, #ecfdf5, #d1fae5);
          padding: 1.5rem;
          border-radius: 0.75rem;
          margin-bottom: 1.5rem;
          animation: slide-in-fade 0.5s ease;
          box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        }
        
        .success-icon {
          font-size: 2.5rem;
          color: #059669;
          animation: pulse 2s infinite;
        }
        
        .success-message h3 {
          color: #065f46;
          margin-bottom: 0.5rem;
          font-size: 1.5rem;
        }
        
        .success-message p {
          color: #047857;
          margin: 0;
        }
        
        /* Warning banner */
        .warning-banner {
          display: flex;
          align-items: center;
          gap: 1.5rem;
          background: linear-gradient(135deg, #fffbeb, #fef3c7);
          padding: 1.5rem;
          border-radius: 0.75rem;
          margin-bottom: 1.5rem;
          border-left: 4px solid #f59e0b;
          animation: slide-in-fade 0.5s ease;
        }
        
        .warning-icon {
          font-size: 2.5rem;
          color: #d97706;
        }
        
        .warning-message h4 {
          color: #b45309;
          margin: 0 0 0.5rem 0;
          font-size: 1.2rem;
        }
        
        .warning-message p {
          color: #92400e;
          margin: 0;
        }
        
        /* Summary Card */
        .summary-card {
          background: linear-gradient(135deg, #eff6ff, #dbeafe);
          border-radius: 0.75rem;
          overflow: hidden;
          margin-bottom: 1.5rem;
          box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
          border: 1px solid #bfdbfe;
        }
        
        .summary-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 1rem 1.5rem;
          background-color: rgba(59, 130, 246, 0.1);
          cursor: pointer;
          transition: background-color 0.2s;
        }
        
        .summary-header:hover {
          background-color: rgba(59, 130, 246, 0.15);
        }
        
        .summary-title {
          display: flex;
          align-items: center;
          gap: 0.75rem;
          color: #1e40af;
          font-size: 1.35rem;
          font-weight: 600;
        }
        
        .summary-icon {
          font-size: 1.5rem;
        }
        
        .expand-button {
          background: none;
          border: none;
          color: #3b82f6;
          font-size: 0.9rem;
          cursor: pointer;
          transition: color 0.2s;
        }
        
        .expand-button:hover {
          color: #1e40af;
        }
        
        .summary-content {
          padding: 1.5rem;
        }
        
        .summary-section {
          margin-bottom: 1.5rem;
          animation: fade-in 0.5s ease;
        }
        
        .summary-section:last-child {
          margin-bottom: 0;
        }
        
        .summary-section-title {
          color: #2563eb;
          margin: 0 0 0.75rem 0;
          font-size: 1.1rem;
          border-bottom: 1px solid rgba(59, 130, 246, 0.2);
          padding-bottom: 0.5rem;
        }
        
        .summary-item {
          display: flex;
          gap: 0.75rem;
          margin-bottom: 0.5rem;
        }
        
        .bullet-point {
          color: #3b82f6;
          font-size: 1.25rem;
          line-height: 1.5;
        }
        
        .summary-item p {
          margin: 0;
          color: #1e3a8a;
          flex: 1;
          line-height: 1.5;
        }
        
        .show-more-button {
          display: block;
          width: 100%;
          padding: 0.75rem;
          background-color: rgba(59, 130, 246, 0.1);
          border: none;
          border-radius: 0.5rem;
          color: #3b82f6;
          font-size: 0.9rem;
          cursor: pointer;
          transition: background-color 0.2s;
          margin-top: 1rem;
        }
        
        .show-more-button:hover {
          background-color: rgba(59, 130, 246, 0.2);
        }
        
        .preview-tabs {
          display: flex;
          gap: 0.5rem;
          margin-bottom: 1.5rem;
          flex-wrap: wrap;
        }
        
        .tab-button {
          padding: 0.75rem 1.25rem;
          background: white;
          border: 1px solid var(--gray-200);
          border-radius: 0.5rem;
          font-weight: 500;
          color: var(--gray-600);
          cursor: pointer;
          transition: all 0.2s ease;
          display: flex;
          align-items: center;
          gap: 0.5rem;
          flex-grow: 1;
          justify-content: center;
        }
        
        .tab-button:hover {
          background: var(--gray-50);
          color: var(--primary);
          transform: translateY(-2px);
          box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        }
        
        .tab-button.active {
          background: var(--primary-lightest);
          color: var(--primary);
          border-color: var(--primary-light);
          box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
          transform: translateY(-2px);
        }
        
        .tab-icon {
          font-size: 1.25rem;
        }
        
        .preview-content {
          border: 1px solid var(--gray-200);
          border-radius: 0.75rem;
          background-color: white;
          overflow: hidden;
          margin-bottom: 2rem;
          box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        }
        
        /* PDF Preview improvements */
        .pdf-preview {
          height: 700px;
          position: relative;
          display: flex;
          flex-direction: column;
          background-color: #f8fafc;
        }
        
        .pdf-toolbar {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 0.75rem 1.25rem;
          background-color: #1e293b;
          color: white;
          border-bottom: 1px solid #334155;
        }
        
        .zoom-controls {
          display: flex;
          align-items: center;
          gap: 0.5rem;
        }
        
        .zoom-button {
          background-color: rgba(255, 255, 255, 0.1);
          color: white;
          border: none;
          width: 2rem;
          height: 2rem;
          border-radius: 4px;
          display: flex;
          align-items: center;
          justify-content: center;
          cursor: pointer;
          transition: background-color 0.2s;
        }
        
        .zoom-button:hover {
          background-color: rgba(255, 255, 255, 0.2);
        }
        
        .zoom-button.reset {
          font-size: 1.2rem;
        }
        
        .zoom-level {
          font-size: 0.9rem;
          min-width: 3rem;
          text-align: center;
        }
        
        .pdf-info {
          font-size: 0.85rem;
          color: #cbd5e1;
        }
        
        .pdf-method strong {
          color: white;
        }
        
        .pdf-viewer-container {
          flex: 1;
          overflow: auto;
          transition: transform 0.3s ease;
          transform-origin: top center;
        }
        
        .pdf-error {
          padding: 2rem;
          display: flex;
          align-items: flex-start;
          gap: 1.5rem;
          background-color: #fff5f7;
          height: 100%;
          animation: fade-in 0.5s ease;
        }
        
        .error-icon {
          font-size: 2rem;
          color: #e11d48;
        }
        
        .error-content h4 {
          color: #be123c;
          margin-top: 0;
          margin-bottom: 0.75rem;
          font-size: 1.25rem;
        }
        
        .error-content p {
          color: #9f1239;
          margin-bottom: 0.75rem;
        }
        
        .error-details {
          background-color: #fecdd3;
          padding: 1rem;
          border-radius: 0.5rem;
          font-family: monospace;
          font-size: 0.9rem;
          margin-top: 1rem;
          color: #881337;
          overflow-x: auto;
        }
        
        /* LaTeX Code improvements */
        .latex-preview {
          height: 700px;
          display: flex;
          flex-direction: column;
        }
        
        .code-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 0.75rem 1.25rem;
          background-color: #1f2937;
          color: white;
        }
        
        .code-title {
          font-weight: 500;
        }
        
        .copy-button {
          background-color: rgba(255, 255, 255, 0.1);
          color: white;
          border: none;
          padding: 0.5rem 1rem;
          border-radius: 0.25rem;
          font-size: 0.85rem;
          cursor: pointer;
          transition: all 0.2s;
          display: flex;
          align-items: center;
          gap: 0.5rem;
        }
        
        .copy-button:hover {
          background-color: rgba(255, 255, 255, 0.2);
        }
        
        .copy-button.copied {
          background-color: #10b981;
        }
        
        .copy-button.copied:before {
          content: '';
          display: inline-block;
          width: 0.8rem;
          height: 0.8rem;
          border-radius: 50%;
          background-color: white;
        }
        
        .latex-code {
          padding: 1.25rem;
          font-family: 'Fira Code', monospace;
          background-color: #111827;
          color: #e5e7eb;
          overflow: auto;
          flex: 1;
          margin: 0;
          line-height: 1.6;
          font-size: 0.9rem;
        }
        
        /* Download buttons improvements */
        .download-options {
          display: flex;
          gap: 1.5rem;
          margin-bottom: 2rem;
        }
        
        .download-button {
          flex: 1;
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 0.75rem;
          padding: 1rem;
          border-radius: 0.75rem;
          border: none;
          font-weight: 500;
          font-size: 1rem;
          cursor: pointer;
          transition: all 0.3s ease;
          position: relative;
          overflow: hidden;
        }
        
        .download-button:before {
          content: '';
          position: absolute;
          top: 0;
          left: 0;
          width: 100%;
          height: 100%;
          background: linear-gradient(45deg, rgba(255,255,255,0) 0%, rgba(255,255,255,0.2) 50%, rgba(255,255,255,0) 100%);
          transform: translateX(-100%);
          transition: transform 0.6s;
        }
        
        .download-button:hover:before {
          transform: translateX(100%);
        }
        
        .latex-download {
          background: linear-gradient(135deg, #2563eb, #1d4ed8);
          color: white;
          box-shadow: 0 4px 10px rgba(37, 99, 235, 0.3);
        }
        
        .latex-download:hover {
          transform: translateY(-2px);
          box-shadow: 0 6px 15px rgba(37, 99, 235, 0.4);
        }
        
        .pdf-download {
          background: linear-gradient(135deg, #059669, #047857);
          color: white;
          box-shadow: 0 4px 10px rgba(5, 150, 105, 0.3);
        }
        
        .pdf-download:hover {
          transform: translateY(-2px);
          box-shadow: 0 6px 15px rgba(5, 150, 105, 0.4);
        }
        
        .download-button.downloading {
          animation: pulse 1.5s infinite;
        }
        
        .spinner {
          width: 1.2rem;
          height: 1.2rem;
          border: 2px solid rgba(255, 255, 255, 0.3);
          border-radius: 50%;
          border-top-color: white;
          animation: spin 1s linear infinite;
          margin-right: 0.5rem;
        }
        
        .download-button:disabled {
          opacity: 0.6;
          cursor: not-allowed;
          transform: none;
          box-shadow: none;
        }
        
        .download-icon {
          font-size: 1.5rem;
        }
        
        /* Action buttons improvements */
        .action-buttons {
          display: flex;
          justify-content: space-between;
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
        
        .btn-secondary {
          background-color: #f1f5f9;
          color: #334155;
          border: 1px solid #e2e8f0;
        }
        
        .btn-secondary:hover {
          background-color: #e2e8f0;
          transform: translateY(-2px);
          box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        .btn-primary {
          background: linear-gradient(135deg, #3b82f6, #2563eb);
          color: white;
          box-shadow: 0 4px 6px rgba(59, 130, 246, 0.3);
        }
        
        .btn-primary:hover {
          transform: translateY(-2px);
          box-shadow: 0 8px 15px rgba(59, 130, 246, 0.4);
          filter: brightness(1.05);
        }
        
        .button-icon {
          font-size: 1.1rem;
        }
        
        /* Animations */
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.7; }
        }
        
        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
        
        @keyframes slide-in-fade {
          from { opacity: 0; transform: translateY(-10px); }
          to { opacity: 1; transform: translateY(0); }
        }
        
        @keyframes fade-in {
          from { opacity: 0; }
          to { opacity: 1; }
        }
        
        /* Responsive styles */
        @media (max-width: 768px) {
          .download-options {
            flex-direction: column;
          }
          
          .action-buttons {
            flex-direction: column;
            gap: 1rem;
          }
          
          .pdf-preview, .latex-preview, .debug-view {
            height: 500px;
          }
          
          .tab-button {
            padding: 0.6rem 1rem;
            font-size: 0.9rem;
          }
        }
      `}</style>
    </div>
  );
};

export default ResultPreview;