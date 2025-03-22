import React, { useState, useEffect, useRef } from 'react';
import * as pdfjsLib from 'pdfjs-dist';

// Make sure this matches the version in pdfUtils.js
pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';

const PDFViewer = ({ pdfData, scale = 1.0 }) => {
  const canvasRef = useRef(null);
  const iframeRef = useRef(null);
  const [currentPage, setCurrentPage] = useState(1);
  const [numPages, setNumPages] = useState(0);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [pdfDoc, setPdfDoc] = useState(null);
  const [debugInfo, setDebugInfo] = useState(null);
  const [renderMethod, setRenderMethod] = useState('canvas'); // 'canvas' or 'iframe'

  // Load the PDF when pdfData changes
  useEffect(() => {
    console.log("PDFViewer: pdfData changed");
    
    if (!pdfData) {
      setIsLoading(false);
      setError("No PDF data provided");
      setDebugInfo({ 
        type: 'error', 
        message: 'No PDF data provided',
        resultCheck: {
          pdfDataExists: !!pdfData,
          pdfDataType: typeof pdfData,
        }
      });
      return;
    }

    const loadPdf = async () => {
      try {
        setIsLoading(true);
        setError(null);
        
        // Try iframe method first if it's a data URL
        if (typeof pdfData === 'string' && pdfData.startsWith('data:application/pdf;base64,')) {
          console.log("PDFViewer: Using iframe method for data URL");
          setRenderMethod('iframe');
          setIsLoading(false);
          setDebugInfo({ 
            status: 'success', 
            renderMethod: 'iframe',
            dataInfo: { type: 'data URL', length: pdfData.length }
          });
          return; // No need to continue with PDF.js loading
        }

        console.log("PDFViewer: Using PDF.js method");
        setRenderMethod('canvas');

        // Log debugging info about the pdfData
        console.log("PDF data type:", typeof pdfData);
        if (typeof pdfData === 'string') {
          console.log("PDF data starts with:", pdfData.substring(0, 50));
        }

        // Determine if pdfData is a base64 string, URL, or binary data
        let loadingTask;
        let dataInfo = { type: 'unknown', length: 'unknown' };

        if (typeof pdfData === 'string') {
          if (pdfData.startsWith('data:application/pdf;base64,')) {
            // Base64 data URL
            const base64 = pdfData.replace(/^data:application\/pdf;base64,/, '');
            try {
              const binary = atob(base64);
              const len = binary.length;
              const bytes = new Uint8Array(len);
              for (let i = 0; i < len; i++) {
                bytes[i] = binary.charCodeAt(i);
              }
              loadingTask = pdfjsLib.getDocument({ data: bytes.buffer });
              dataInfo = { type: 'base64', length: bytes.length };
            } catch (err) {
              console.error("Error decoding base64:", err);
              throw new Error(`Failed to decode base64: ${err.message}`);
            }
          } else {
            // URL
            loadingTask = pdfjsLib.getDocument({ url: pdfData });
            dataInfo = { type: 'url', url: pdfData };
          }
        } else {
          // ArrayBuffer or TypedArray
          loadingTask = pdfjsLib.getDocument({ data: pdfData });
          dataInfo = { 
            type: 'binary', 
            length: pdfData.byteLength || 'unknown size',
            arrayType: pdfData.constructor.name
          };
        }

        setDebugInfo({ status: 'loading', dataInfo });

        // Load the document
        const pdf = await loadingTask.promise;
        setPdfDoc(pdf);
        setNumPages(pdf.numPages);
        setCurrentPage(1);
        setDebugInfo({ 
          status: 'success', 
          pages: pdf.numPages, 
          dataInfo 
        });
      } catch (err) {
        console.error("Error loading PDF:", err);
        setError(`Failed to load PDF: ${err.message}`);
        setDebugInfo({ 
          status: 'error', 
          message: err.message, 
          stack: err.stack,
          pdfDataType: typeof pdfData
        });
        
        // Fallback to iframe if canvas method failed
        if (typeof pdfData === 'string' && pdfData.startsWith('data:application/pdf;base64,')) {
          console.log("PDFViewer: Falling back to iframe method after PDF.js error");
          setRenderMethod('iframe');
          setError(null); // Clear the error since we're trying a fallback
        }
      } finally {
        setIsLoading(false);
      }
    };

    loadPdf();

    // Cleanup function
    return () => {
      if (pdfDoc) {
        pdfDoc.destroy();
        setPdfDoc(null);
      }
    };
  }, [pdfData]);

  // Render the current page when it changes (only for canvas method)
  useEffect(() => {
    if (renderMethod !== 'canvas' || !pdfDoc || !canvasRef.current) return;

    const renderPage = async () => {
      try {
        setIsLoading(true);
        
        // Get the page
        const page = await pdfDoc.getPage(currentPage);
        
        // Calculate scale to fit the canvas width
        const viewport = page.getViewport({ scale });
        
        // Prepare canvas
        const canvas = canvasRef.current;
        const context = canvas.getContext('2d');
        canvas.height = viewport.height;
        canvas.width = viewport.width;
        
        // Render PDF page
        const renderContext = {
          canvasContext: context,
          viewport: viewport
        };
        
        await page.render(renderContext).promise;
        setDebugInfo(prev => ({ ...prev, renderStatus: 'success' }));
      } catch (err) {
        console.error("Error rendering PDF page:", err);
        setError(`Failed to render page ${currentPage}: ${err.message}`);
        setDebugInfo(prev => ({ 
          ...prev, 
          renderStatus: 'error', 
          renderError: err.message 
        }));
      } finally {
        setIsLoading(false);
      }
    };

    renderPage();
  }, [pdfDoc, currentPage, scale, renderMethod]);

  // Handle page navigation (only for canvas method)
  const goToPreviousPage = () => {
    if (currentPage > 1) {
      setCurrentPage(currentPage - 1);
    }
  };

  const goToNextPage = () => {
    if (currentPage < numPages) {
      setCurrentPage(currentPage + 1);
    }
  };

  return (
    <div className="pdf-viewer">
      {isLoading && <div className="loading">Loading PDF...</div>}
      
      {error && (
        <div className="error-message">
          <p>{error}</p>
          <p>If you're experiencing issues, try:</p>
          <ul>
            <li>Using a different browser (Chrome or Firefox recommended)</li>
            <li>Checking if the PDF is valid and not password protected</li>
            <li>Converting your file to a different format before uploading</li>
          </ul>
          <button 
            onClick={() => setDebugInfo(prev => ({ ...prev, showDetails: !prev?.showDetails }))}
            className="debug-toggle"
          >
            {debugInfo?.showDetails ? 'Hide' : 'Show'} Technical Details
          </button>
          
          {debugInfo?.showDetails && (
            <pre className="debug-info">
              {JSON.stringify(debugInfo, null, 2)}
            </pre>
          )}
        </div>
      )}
      
      {/* Canvas-based PDF rendering */}
      {renderMethod === 'canvas' && (
        <>
          <canvas ref={canvasRef} className="pdf-canvas" />
          
          {numPages > 1 && (
            <div className="pdf-controls">
              <button 
                onClick={goToPreviousPage} 
                disabled={currentPage <= 1 || isLoading}
              >
                Previous
              </button>
              
              <span className="page-info">
                {currentPage} / {numPages}
              </span>
              
              <button 
                onClick={goToNextPage} 
                disabled={currentPage >= numPages || isLoading}
              >
                Next
              </button>
            </div>
          )}
        </>
      )}
      
      {/* Iframe-based PDF rendering */}
      {renderMethod === 'iframe' && (
        <iframe 
          ref={iframeRef}
          src={pdfData}
          className="pdf-iframe"
          title="PDF Preview"
        />
      )}
    </div>
  );
};

export default PDFViewer;