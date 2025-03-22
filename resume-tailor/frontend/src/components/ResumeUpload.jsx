import React, { useState, useRef, useEffect } from 'react';
import { testPdfJs, extractTextFromPdf } from '../utils/pdfUtils';

const ResumeUpload = ({ onUpload, resumeData }) => {
  const [text, setText] = useState(resumeData || '');
  const [dragActive, setDragActive] = useState(false);
  const [uploadStatus, setUploadStatus] = useState({ type: '', message: '' });
  const [isPdfSupported, setIsPdfSupported] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [pdfBase64, setPdfBase64] = useState(null);
  const [uploadAnimation, setUploadAnimation] = useState(false);
  const fileInputRef = useRef(null);
  const dropAreaRef = useRef(null);

  useEffect(() => {
    // Test if PDF.js is working
    const checkPdfSupport = async () => {
      try {
        const result = await testPdfJs();
        setIsPdfSupported(result);
        if (!result) {
          console.warn('PDF.js is not working properly. PDF uploads may not work correctly.');
        }
      } catch (error) {
        console.error('Error testing PDF.js:', error);
        setIsPdfSupported(false);
      }
    };

    checkPdfSupport();
  }, []);

  const handleFileChange = async (e) => {
    const files = e.target.files || e.dataTransfer.files;
    
    if (!files || files.length === 0) {
      return;
    }
    
    const file = files[0];
    setIsProcessing(true);
    setUploadStatus({ type: 'info', message: 'Processing file...' });
    setUploadAnimation(true);
    
    try {
      // If it's a PDF, extract text AND keep the base64 content
      if (file.type === 'application/pdf') {
        const fileText = await extractTextFromPdf(await file.arrayBuffer());
        setText(fileText);
        
        // Also store the PDF as base64 for the backend
        const reader = new FileReader();
        reader.onload = (event) => {
          const base64 = event.target.result;
          setPdfBase64(base64);
          
          // Send both the text and base64 to parent
          onUpload(fileText, base64);
          
          setUploadStatus({ 
            type: 'success', 
            message: 'PDF processed successfully! Both text and original PDF will be used.' 
          });
          setTimeout(() => setUploadAnimation(false), 500);
        };
        reader.readAsDataURL(file);
      } else {
        // For non-PDF files, just extract the text
        const fileText = await extractFileText(file);
        setText(fileText);
        setPdfBase64(null);
        onUpload(fileText, null);
        setUploadStatus({ type: 'success', message: 'File successfully processed!' });
        setTimeout(() => setUploadAnimation(false), 500);
      }
    } catch (error) {
      console.error('Error processing file:', error);
      setUploadStatus({ 
        type: 'error', 
        message: `Error processing file: ${error.message}. Try a different format or paste text directly.` 
      });
      setUploadAnimation(false);
    } finally {
      setIsProcessing(false);
    }
  };

  const extractFileText = async (file) => {
    const fileType = file.type.toLowerCase();
    
    // Handle text files
    if (fileType === 'text/plain' || 
        file.name.endsWith('.txt') || 
        file.name.endsWith('.md') || 
        file.name.endsWith('.rtf')) {
      return await file.text();
    }
    
    // Handle Word documents (this will just suggest conversion)
    if (fileType === 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' ||
        fileType === 'application/msword') {
      throw new Error('Word documents (.doc, .docx) cannot be processed directly. Please copy and paste the content manually.');
    }
    
    throw new Error(`Unsupported file type: ${fileType}`);
  };

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    handleFileChange(e);
  };

  const handleTextChange = (e) => {
    const newText = e.target.value;
    setText(newText);
    setPdfBase64(null); // Clear any PDF data when text is manually changed
    onUpload(newText, null);
  };

  const handleButtonClick = () => {
    fileInputRef.current?.click();
  };
  
  const getFileTypeIcon = (fileName) => {
    if (!fileName) return '📄';
    
    const extension = fileName.split('.').pop()?.toLowerCase();
    
    switch(extension) {
      case 'pdf': return '📕';
      case 'doc':
      case 'docx': return '📘';
      case 'txt': return '📃';
      case 'rtf': return '📝';
      default: return '📄';
    }
  };

  return (
    <div className="resume-upload">
      <h2 className="section-title">Upload Your Resume</h2>
      <p className="section-description">
        Start by uploading your existing resume or paste its content below. We'll analyze it and tailor it to match the job description.
      </p>
      
      <div className="upload-methods">
        <div 
          ref={dropAreaRef}
          className={`upload-area ${dragActive ? 'dragging' : ''} ${uploadAnimation ? 'file-flying' : ''} ${isProcessing ? 'processing' : ''}`}
          onDragEnter={handleDrag}
          onDragOver={handleDrag}
          onDragLeave={handleDrag}
          onDrop={handleDrop}
          onClick={handleButtonClick}
        >
          {isProcessing ? (
            <div className="processing-indicator">
              <div className="spinner"></div>
              <p className="processing-text">Processing your resume...</p>
            </div>
          ) : (
            <>
              <div className="upload-icon">{uploadAnimation ? getFileTypeIcon(fileInputRef.current?.files?.[0]?.name) : '📄'}</div>
              <p className="upload-text">
                {dragActive ? 'Drop your resume here' : 'Drag & drop your resume file here'}
              </p>
              <div className="upload-separator">or</div>
              <button 
                className="btn btn-primary upload-btn" 
                onClick={(e) => {
                  e.stopPropagation();
                  handleButtonClick();
                }}
                disabled={isProcessing}
              >
                Browse Files
              </button>
              <p className="upload-help">
                Supports PDF, TXT, RTF, and MD files
              </p>
            </>
          )}
          
          <input
            ref={fileInputRef}
            type="file"
            className="hidden-input"
            onChange={handleFileChange}
            accept=".pdf,.txt,.rtf,.md,.doc,.docx"
            disabled={isProcessing}
          />
          
          {isPdfSupported === false && (
            <div className="warning-message">
              ⚠️ PDF processing may not work in your browser. Try a different file format or paste text directly.
            </div>
          )}
        </div>

        {uploadStatus.message && (
          <div className={`status-message ${uploadStatus.type}`}>
            {uploadStatus.type === 'error' ? '❌ ' : 
             uploadStatus.type === 'success' ? '✅ ' : 
             uploadStatus.type === 'info' ? 'ℹ️ ' : ''}
            {uploadStatus.message}
          </div>
        )}
        
        <div className="divider">
          <span className="divider-text">or paste your resume text</span>
        </div>
        
        <div className="textarea-container">
          <textarea
            className="resume-textarea"
            value={text}
            onChange={handleTextChange}
            placeholder="Paste your resume text here..."
            rows={15}
          />
          
          {text && (
            <div className="character-count">
              {text.length} characters
            </div>
          )}
        </div>
      </div>
      
      <div className="action-buttons">
        <button 
          className="btn btn-primary" 
          onClick={() => onUpload(text, pdfBase64)}
          disabled={isProcessing || !text.trim()}
        >
          Continue
          <span className="button-icon">→</span>
        </button>
      </div>
      
      {/* Add styles */}
      <style jsx>{`
        .resume-upload {
          max-width: 800px;
          margin: 0 auto;
        }
        
        .section-description {
          color: var(--gray-600);
          margin-bottom: 2rem;
          text-align: center;
          max-width: 600px;
          margin-left: auto;
          margin-right: auto;
        }
        
        .upload-methods {
          display: flex;
          flex-direction: column;
          gap: 2rem;
        }
        
        .upload-area {
          border: 2px dashed var(--gray-300);
          border-radius: 1rem;
          padding: 3rem 2rem;
          text-align: center;
          transition: all 0.3s ease;
          background-color: var(--gray-50);
          cursor: pointer;
          position: relative;
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          min-height: 300px;
        }
        
        .upload-area:hover, .upload-area.dragging {
          border-color: var(--primary);
          background-color: var(--primary-lightest);
        }
        
        .upload-area.processing {
          cursor: wait;
          border-color: var(--primary);
          background-color: var(--primary-lightest);
        }
        
        .upload-icon {
          font-size: 4rem;
          margin-bottom: 1.5rem;
          transition: all 0.3s ease;
        }
        
        .upload-area:hover .upload-icon {
          transform: translateY(-10px);
        }
        
        .upload-area.file-flying .upload-icon {
          animation: fly-in 0.5s ease;
        }
        
        @keyframes fly-in {
          0% { transform: translateY(-100px) scale(0.5); opacity: 0; }
          70% { transform: translateY(10px) scale(1.1); }
          100% { transform: translateY(0) scale(1); opacity: 1; }
        }
        
        .upload-text {
          margin-bottom: 1rem;
          color: var(--gray-700);
          font-weight: 600;
          font-size: 1.25rem;
        }
        
        .upload-separator {
          display: flex;
          align-items: center;
          margin: 1rem 0;
          color: var(--gray-500);
          font-size: 0.9rem;
        }
        
        .upload-separator:before,
        .upload-separator:after {
          content: "";
          flex: 1;
          border-bottom: 1px solid var(--gray-300);
          margin: 0 10px;
        }
        
        .upload-btn {
          padding: 0.75rem 2rem;
          margin-bottom: 1rem;
        }
        
        .upload-help {
          font-size: 0.85rem;
          color: var(--gray-500);
        }
        
        .hidden-input {
          display: none;
        }
        
        .divider {
          display: flex;
          align-items: center;
          margin: 1rem 0;
        }
        
        .divider:before,
        .divider:after {
          content: "";
          flex: 1;
          border-bottom: 1px solid var(--gray-300);
        }
        
        .divider-text {
          padding: 0 1rem;
          color: var(--gray-500);
          font-size: 0.9rem;
          font-weight: 500;
        }
        
        .textarea-container {
          position: relative;
        }
        
        .resume-textarea {
          width: 100%;
          padding: 1rem;
          border: 1px solid var(--gray-300);
          border-radius: 0.75rem;
          font-family: inherit;
          font-size: 0.95rem;
          line-height: 1.6;
          min-height: 250px;
          resize: vertical;
          transition: all 0.2s ease;
          background-color: white;
        }
        
        .resume-textarea:focus {
          outline: none;
          border-color: var(--primary);
          box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.15);
        }
        
        .character-count {
          position: absolute;
          bottom: 0.5rem;
          right: 0.75rem;
          font-size: 0.8rem;
          color: var(--gray-400);
        }
        
        .action-buttons {
          display: flex;
          justify-content: flex-end;
          margin-top: 2rem;
        }
        
        .button-icon {
          margin-left: 0.5rem;
          font-size: 1.1rem;
          transition: transform 0.2s ease;
        }
        
        .btn:hover .button-icon {
          transform: translateX(3px);
        }
        
        .status-message {
          padding: 0.75rem 1rem;
          border-radius: 0.5rem;
          font-size: 0.9rem;
          animation: fade-in 0.3s ease;
        }
        
        .status-message.success {
          background-color: #ecfdf5;
          color: #065f46;
          border-left: 3px solid #10b981;
        }
        
        .status-message.error {
          background-color: #fef2f2;
          color: #b91c1c;
          border-left: 3px solid #ef4444;
        }
        
        .status-message.info {
          background-color: #eff6ff;
          color: #1e40af;
          border-left: 3px solid #3b82f6;
        }
        
        @keyframes fade-in {
          from { opacity: 0; transform: translateY(-10px); }
          to { opacity: 1; transform: translateY(0); }
        }
        
        .processing-indicator {
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: 1.5rem;
        }
        
        .spinner {
          width: 50px;
          height: 50px;
          border: 4px solid rgba(59, 130, 246, 0.2);
          border-radius: 50%;
          border-top-color: var(--primary);
          animation: spin 1s linear infinite;
        }
        
        .processing-text {
          color: var(--primary);
          font-weight: 500;
        }
        
        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
        
        .warning-message {
          margin-top: 1rem;
          padding: 0.75rem;
          background-color: #fff7ed;
          color: #9a3412;
          border-radius: 0.5rem;
          font-size: 0.9rem;
          display: flex;
          align-items: center;
          gap: 0.5rem;
          border-left: 3px solid #f97316;
        }
      `}</style>
    </div>
  );
};

export default ResumeUpload;