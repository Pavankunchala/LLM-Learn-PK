/**
 * Utility functions for downloading PDFs and LaTeX files
 */

/**
 * Download a PDF from base64 data
 * @param {string} base64Data - Base64-encoded PDF data
 * @param {string} [filename='resume.pdf'] - Name for the downloaded file
 * @returns {boolean} Success status
 */
export const downloadPdfFromBase64 = (base64Data, filename = 'resume.pdf') => {
    try {
      console.log("Starting PDF download...");
      console.log("Base64 data length:", base64Data?.length || 0);
      
      // Check if base64 data is valid
      if (!base64Data || typeof base64Data !== 'string') {
        console.error("Invalid base64 data type:", typeof base64Data);
        return false;
      }
  
      // Remove potential data URL prefix
      let cleanBase64 = base64Data;
      if (base64Data.includes('base64,')) {
        cleanBase64 = base64Data.split('base64,')[1];
        console.log("Removed data URL prefix, new length:", cleanBase64.length);
      }
      
      try {
        // Test decode a small portion to validate the base64
        const testDecode = atob(cleanBase64.substring(0, 10));
        console.log("Base64 validation successful", testDecode.length);
      } catch (err) {
        console.error("Base64 validation failed:", err);
        return false;
      }
      
      // Convert base64 to binary data
      let byteCharacters;
      try {
        byteCharacters = atob(cleanBase64);
        console.log("Base64 decoded successfully, binary length:", byteCharacters.length);
      } catch (err) {
        console.error("Error decoding base64:", err);
        return false;
      }
      
      const byteArrays = [];
      
      // Split the binary data into chunks
      for (let offset = 0; offset < byteCharacters.length; offset += 512) {
        const slice = byteCharacters.slice(offset, offset + 512);
        
        const byteNumbers = new Array(slice.length);
        for (let i = 0; i < slice.length; i++) {
          byteNumbers[i] = slice.charCodeAt(i);
        }
        
        const byteArray = new Uint8Array(byteNumbers);
        byteArrays.push(byteArray);
      }
      
      console.log("Created byte arrays:", byteArrays.length);
      
      // Create blob and download link
      const blob = new Blob(byteArrays, { type: 'application/pdf' });
      const url = URL.createObjectURL(blob);
      
      console.log("Created blob URL:", url);
      console.log("Blob size:", blob.size, "bytes");
      
      // Test if the blob is valid
      if (blob.size === 0) {
        console.error("Generated PDF blob is empty");
        return false;
      }
      
      // Create and trigger download link
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      
      // Clean up
      setTimeout(() => {
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        console.log("PDF download cleanup complete");
      }, 100);
      
      return true;
    } catch (error) {
      console.error("Error downloading PDF:", error);
      return false;
    }
  };
  
  /**
   * Download text content (like LaTeX) as a file
   * @param {string} content - The text content to download
   * @param {string} [filename='resume.tex'] - Name for the downloaded file
   * @param {string} [mimeType='text/plain'] - MIME type of the content
   * @returns {boolean} Success status
   */
  export const downloadTextContent = (content, filename = 'resume.tex', mimeType = 'text/plain') => {
    try {
      console.log("Starting text content download...");
      
      if (!content) {
        console.error("No content provided for download");
        return false;
      }
      
      const blob = new Blob([content], { type: mimeType });
      const url = URL.createObjectURL(blob);
      
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      
      // Clean up
      setTimeout(() => {
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        console.log("Text download cleanup complete");
      }, 100);
      
      return true;
    } catch (error) {
      console.error("Error downloading text content:", error);
      return false;
    }
  };
  
  /**
   * Validate base64 data to check if it's properly formatted
   * @param {string} base64String - The base64 string to validate
   * @returns {boolean} True if the string is valid base64
   */
  export const isValidBase64 = (base64String) => {
    if (!base64String || typeof base64String !== 'string') {
      return false;
    }
    
    // Remove potential data URL prefix
    let cleanBase64 = base64String;
    if (base64String.includes('base64,')) {
      cleanBase64 = base64String.split('base64,')[1];
    }
    
    // Check for valid base64 pattern (optional for quick validation)
    const base64Pattern = /^[A-Za-z0-9+/=]+$/;
    if (!base64Pattern.test(cleanBase64)) {
      console.warn("Base64 string contains invalid characters");
      // We'll still try to decode it below, as some valid base64 strings might
      // not strictly match this pattern (especially if they contain padding)
    }
    
    try {
      // Check if it can be decoded (this is the definitive test)
      // Only try to decode a small portion to avoid performance issues
      // with very large strings
      const testLength = Math.min(cleanBase64.length, 100);
      window.atob(cleanBase64.substring(0, testLength));
      return true;
    } catch (e) {
      console.error("Invalid base64 format:", e);
      return false;
    }
  };
  
  /**
   * Debug utility to get information about PDF data
   * @param {string|ArrayBuffer} pdfData - The PDF data to analyze
   * @returns {Object} Debug information
   */
  export const analyzePdfData = (pdfData) => {
    const info = {
      type: typeof pdfData,
      valid: false,
      details: {}
    };
    
    if (!pdfData) {
      info.details.error = "No PDF data provided";
      return info;
    }
    
    if (typeof pdfData === 'string') {
      info.details.length = pdfData.length;
      info.details.start = pdfData.substring(0, 50);
      
      if (pdfData.startsWith('data:application/pdf;base64,')) {
        info.details.format = "data URL (base64)";
        const base64 = pdfData.replace(/^data:application\/pdf;base64,/, '');
        info.details.base64Length = base64.length;
        info.valid = isValidBase64(base64);
        
        // Additional validation: try to decode a small portion
        try {
          const testDecode = atob(base64.substring(0, 10));
          info.details.testDecodeLength = testDecode.length;
          info.details.testDecodeSuccess = true;
        } catch (err) {
          info.details.testDecodeSuccess = false;
          info.details.testDecodeError = err.message;
        }
      } else if (pdfData.startsWith('%PDF-')) {
        info.details.format = "PDF plain text";
        info.valid = true;
      } else if (isValidBase64(pdfData)) {
        info.details.format = "base64 without data URL prefix";
        info.valid = true;
        
        // Additional validation
        try {
          const testDecode = atob(pdfData.substring(0, 10));
          info.details.testDecodeLength = testDecode.length;
          info.details.testDecodeSuccess = true;
        } catch (err) {
          info.details.testDecodeSuccess = false;
          info.details.testDecodeError = err.message;
        }
      } else {
        info.details.format = "unknown string format";
        
        // Try to determine if it's perhaps JSON or some other format
        if (pdfData.startsWith('{') && pdfData.endsWith('}')) {
          info.details.possibleFormat = "JSON string";
          try {
            const parsed = JSON.parse(pdfData);
            info.details.jsonParsed = true;
            info.details.jsonKeys = Object.keys(parsed);
          } catch (e) {
            info.details.jsonParsed = false;
          }
        } else if (pdfData.startsWith('<') && pdfData.endsWith('>')) {
          info.details.possibleFormat = "XML/HTML string";
        }
      }
    } else if (pdfData instanceof ArrayBuffer) {
      info.details.format = "ArrayBuffer";
      info.details.byteLength = pdfData.byteLength;
      info.valid = pdfData.byteLength > 0;
      
      // Additional check for PDF signature
      try {
        const firstBytes = new Uint8Array(pdfData, 0, 5);
        const signature = String.fromCharCode.apply(null, firstBytes);
        info.details.startsWithPdfSignature = signature === '%PDF-';
      } catch (e) {
        info.details.signatureCheckError = e.message;
      }
    } else if (pdfData instanceof Uint8Array) {
      info.details.format = "Uint8Array";
      info.details.length = pdfData.length;
      info.valid = pdfData.length > 0;
      
      // Check for PDF signature
      if (pdfData.length >= 5) {
        const signature = String.fromCharCode(
          pdfData[0], pdfData[1], pdfData[2], pdfData[3], pdfData[4]
        );
        info.details.startsWithPdfSignature = signature === '%PDF-';
      }
    } else {
      info.details.format = "unknown type";
      info.details.constructorName = pdfData?.constructor?.name;
    }
    
    return info;
  };