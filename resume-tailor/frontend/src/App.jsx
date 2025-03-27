import React, { useState, useEffect } from 'react';
import Header from './components/Header';
import Footer from './components/Footer';
import ResumeUpload from './components/ResumeUpload';
import JobDescription from './components/JobDescription';
import ModelSelection from './components/ModelSelection';
import ResultPreview from './components/ResultPreview';
import ProcessingOverlay from './components/ProcessingOverlay';

const App = () => {
  // State management
  const [resumeData, setResumeData] = useState('');
  const [resumePdfBase64, setResumePdfBase64] = useState(null);
  const [resumeFormat, setResumeFormat] = useState('txt');
  const [jobDescription, setJobDescription] = useState('');
  const [selectedModel, setSelectedModel] = useState('llama3');
  const [isProcessing, setIsProcessing] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [activeStep, setActiveStep] = useState(1);
  const [taskId, setTaskId] = useState(null);
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState('');
  
  // Set the selected model in the window for the ProcessingOverlay to access
  useEffect(() => {
    if (selectedModel) {
      window.selectedModel = selectedModel;
    }
  }, [selectedModel]);
  
  // Poll for task status when processing
  useEffect(() => {
    let intervalId;
    
    if (isProcessing && taskId) {
      intervalId = setInterval(pollTaskStatus, 2000);
    }
    
    return () => {
      if (intervalId) clearInterval(intervalId);
    };
  }, [isProcessing, taskId]);
  
  // Function to poll for task status
  const pollTaskStatus = async () => {
    try {
      const response = await fetch('http://localhost:5000/process/status');
      if (!response.ok) {
        throw new Error(`Status check failed with status: ${response.status}`);
      }
      
      const data = await response.json();
      console.log("Task status:", data);
      
      // Update progress and status
      setProgress(data.progress || 0);
      setStatus(data.status || '');
      
      // If task is completed or errored, fetch the result
      if (data.status === 'completed' || data.status === 'error') {
        if (data.status === 'error') {
          console.error("Task error:", data.error);
        }
        fetchResult();
      }
    } catch (error) {
      console.error("Error polling status:", error);
      // Don't stop polling on error, keep trying
    }
  };
  
  // Function to fetch the final result
  const fetchResult = async () => {
    if (!taskId) return;
    
    try {
      const response = await fetch(`http://localhost:5000/process/result/${taskId}`);
      if (!response.ok) {
        if (response.status === 404) {
          console.log("Result not ready yet, continuing to poll");
          return; // Continue polling if result not ready
        }
        throw new Error(`Failed to fetch result: ${response.status}`);
      }
      
      const data = await response.json();
      console.log("Result data received:", {
        status: data.status,
        hasLatex: !!data.latex,
        hasPdf: !!data.pdf,
        latexLength: data.latex?.length || 0,
        pdfLength: data.pdf?.length || 0
      });
      
      // Only process the result if it has both LaTeX and either PDF or an error
      if (data.latex && (data.pdf || data.status === 'error')) {
        setResult(data);
        setActiveStep(4); // Move to result step
        // Wait a moment before stopping the processing overlay
        setTimeout(() => {
          setIsProcessing(false);
          setTaskId(null);
        }, 1500);
      } else {
        console.log("Incomplete result, continuing to poll", data);
      }
    } catch (error) {
      console.error("Error fetching result:", error);
      setError(`Failed to retrieve result: ${error.message}`);
      // Keep processing state active to show error in overlay
    }
  };
  
  // Handle resume upload
  const handleResumeUpload = (text, pdfBase64, format = 'txt') => {
    setResumeData(text);
    setResumePdfBase64(pdfBase64);
    setResumeFormat(format);
    setActiveStep(2);
  };

  // Handle job description change
  const handleJobDescriptionChange = (text) => {
    setJobDescription(text);
  };

  // Handle model change
  const handleModelChange = (model) => {
    setSelectedModel(model);
  };
  
  // Function to start processing
  const processResume = async () => {
    if (!resumeData || !jobDescription) {
      setError('Please provide both a resume and job description');
      return;
    }
  
    setIsProcessing(true);
    setError(null);
    setProgress(0); // Use your existing variable name
    setStatus('starting'); // Use your existing variable name
  
    try {
      const requestBody = {
        resume: resumePdfBase64 || resumeData,
        resumeFormat: resumePdfBase64 ? 'pdf_base64' : 'text',
        jobDescription,
        model: selectedModel,
      };
      
      // Start the processing job
      const response = await fetch('http://localhost:5000/process', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestBody),
      });
  
      if (!response.ok) throw new Error('Failed to process resume');
      
      const initData = await response.json();
      if (initData.error) throw new Error(initData.error);
      
      // Get the task ID
      const taskId = initData.task_id;
      if (!taskId) throw new Error('No task ID returned from server');
      
      // Continue polling for result
      let isComplete = false;
      const maxAttempts = 180; // Give up after ~2 minutes
      let attempts = 0;
      
      while (!isComplete && attempts < maxAttempts) {
        await new Promise(resolve => setTimeout(resolve, 2000)); // Wait 2 seconds
        attempts++;
        
        try {
          // Poll status endpoint
          const statusResponse = await fetch('http://localhost:5000/process/status');
          if (!statusResponse.ok) continue;
          
          const statusData = await statusResponse.json();
          setProgress(statusData.progress || 0);
          setStatus(statusData.status || '');
          
          // Check if processing is complete
          if (statusData.status === 'completed' || statusData.status === 'error') {
            isComplete = true;
            
            // Fetch the final result
            const resultResponse = await fetch(`http://localhost:5000/process/result/${taskId}`);
            if (!resultResponse.ok) throw new Error('Failed to retrieve result');
            
            const resultData = await resultResponse.json();
            setResult(resultData);
            setActiveStep(4);
          }
        } catch (err) {
          console.error('Error checking status:', err);
          // Continue polling on error
        }
      }
      
      // If we hit max attempts, show error
      if (!isComplete) {
        throw new Error('Processing timed out. Please try again.');
      }
    } catch (err) {
      setError(err.message || 'An error occurred while processing your resume');
    } finally {
      setIsProcessing(false);
    }
  };
  // Function to reset the app
  const resetApp = () => {
    setResumeData('');
    setResumePdfBase64(null);
    setResumeFormat('txt');
    setJobDescription('');
    setSelectedModel('llama3');
    setResult(null);
    setError(null);
    setTaskId(null);
    setProgress(0);
    setStatus('');
    setActiveStep(1);
  };

  // Define steps for the progress indicator
  const steps = [
    { step: 1, label: 'Upload Resume', icon: '📄' },
    { step: 2, label: 'Job Description', icon: '📝' },
    { step: 3, label: 'Select Model', icon: '🤖' },
    { step: 4, label: 'Results', icon: '✅' }
  ];
  
  return (
    <div className="app-container">
      <Header />
      
      <main className="main-content">
        <div className="hero-section">
          <div className="hero-content">
            <h1 className="hero-title">Resume Tailor</h1>
            <p className="hero-subtitle">Customize your resume for specific job postings using AI</p>
            <button className="cta-button" onClick={() => window.scrollTo({
              top: document.querySelector('.process-card').offsetTop - 100, 
              behavior: 'smooth'
            })}>
              Get Started
            </button>

            <style jsx>{`
              .process-icons {
                display: flex;
                align-items: center;
                justify-content: center;
                margin-top: 2rem;
                gap: 0.5rem;
              }
              
              .icon-item {
                display: flex;
                align-items: center;
                justify-content: center;
                width: 50px;
                height: 50px;
                background-color: rgba(255, 255, 255, 0.15);
                backdrop-filter: blur(8px);
                border-radius: 12px;
                font-size: 1.5rem;
                box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
              }
              
              .icon-connector {
                color: rgba(255, 255, 255, 0.7);
                font-size: 1.25rem;
              }
            `}</style>
          </div>
          <div className="process-icons">
            <div className="icon-item">📄</div>
            <div className="icon-connector">➡️</div>
            <div className="icon-item">💼</div>
            <div className="icon-connector">➡️</div>
            <div className="icon-item">✨</div>
            <div className="icon-connector">➡️</div>
            <div className="icon-item">🎯</div>
          </div>
        </div>

        <div className="container">
          <div className="card process-card">
            <div className="progress-container">
              <div className="progress-steps">
                {steps.map(({ step, label, icon }) => (
                  <div 
                    key={step} 
                    className={`progress-step ${step <= activeStep ? 'active' : ''} ${step === activeStep ? 'current' : ''}`}
                    onClick={() => step < activeStep && setActiveStep(step)}
                  >
                    <div className="step-icon">{icon}</div>
                    <div className="step-number">{step}</div>
                    <div className="step-label">{label}</div>
                    {step < 4 && <div className="step-connector"></div>}
                  </div>
                ))}
              </div>
              <div className="progress-bar">
                <div 
                  className="progress-indicator" 
                  style={{ width: `${(activeStep - 1) * 33.33}%` }}
                ></div>
              </div>
            </div>
            
            <div className="step-content">
              {activeStep === 1 && (
                <ResumeUpload 
                  onUpload={handleResumeUpload} 
                  resumeData={resumeData}
                  resumePdfBase64={resumePdfBase64}
                />
              )}
              
              {activeStep === 2 && (
                <JobDescription 
                  value={jobDescription} 
                  onChange={handleJobDescriptionChange} 
                  onBack={() => setActiveStep(1)}
                  onNext={() => setActiveStep(3)}
                  isValid={jobDescription.trim().length > 50}
                />
              )}
              
              {activeStep === 3 && (
                <ModelSelection 
                  selectedModel={selectedModel} 
                  onChange={handleModelChange}
                  onBack={() => setActiveStep(2)}
                  onProcess={processResume}
                  isProcessing={isProcessing}
                />
              )}
              
              {activeStep === 4 && result && (
                <ResultPreview 
                  result={result}
                  onBack={() => setActiveStep(3)}
                  onReset={resetApp}
                />
              )}
            </div>
            
            {error && (
              <div className="error-message">
                <span className="error-icon">⚠️</span>
                {error}
              </div>
            )}
          </div>
          
          <div className="features-section">
            <h2 className="section-title">How It Works</h2>
            <div className="features-grid">
              <div className="feature-card">
                <div className="feature-icon">📄</div>
                <h3 className="feature-title">Upload Your Resume</h3>
                <p className="feature-description">Start by uploading your existing resume or paste its content.</p>
              </div>
              <div className="feature-card">
                <div className="feature-icon">📝</div>
                <h3 className="feature-title">Provide Job Description</h3>
                <p className="feature-description">Enter the job description you're applying for.</p>
              </div>
              <div className="feature-card">
                <div className="feature-icon">🤖</div>
                <h3 className="feature-title">Choose AI Model</h3>
                <p className="feature-description">Select an AI model that fits your needs.</p>
              </div>
              <div className="feature-card">
                <div className="feature-icon">✨</div>
                <h3 className="feature-title">Get Tailored Resume</h3>
                <p className="feature-description">Receive a professional resume optimized for your target job.</p>
              </div>
            </div>
          </div>
          
          <div className="testimonials-section">
            <h2 className="section-title">What Users Say</h2>
            <div className="testimonials-grid">
              <div className="testimonial-card">
                <div className="testimonial-content">
                  "This tool helped me land interviews at 3 companies that previously rejected me. The AI tailoring made all the difference!"
                </div>
                <div className="testimonial-author">
                  <div className="author-avatar">👩‍💼</div>
                  <div className="author-info">
                    <div className="author-name">Sarah J.</div>
                    <div className="author-title">UX Designer</div>
                  </div>
                </div>
              </div>
              <div className="testimonial-card">
                <div className="testimonial-content">
                  "As a career coach, I recommend this to all my clients. It saves hours of manual resume tweaking for each application."
                </div>
                <div className="testimonial-author">
                  <div className="author-avatar">👨‍🏫</div>
                  <div className="author-info">
                    <div className="author-name">Michael T.</div>
                    <div className="author-title">Career Coach</div>
                  </div>
                </div>
              </div>
              <div className="testimonial-card">
                <div className="testimonial-content">
                  "The ATS optimization feature helped my resume actually get seen by hiring managers instead of being filtered out."
                </div>
                <div className="testimonial-author">
                  <div className="author-avatar">👨‍💻</div>
                  <div className="author-info">
                    <div className="author-name">David L.</div>
                    <div className="author-title">Software Engineer</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </main>
      
      <Footer />

      {/* Render ProcessingOverlay over the app when processing */}
      {isProcessing && (
        <ProcessingOverlay 
          isProcessing={isProcessing}
          progress={progress}
          status={status}
          error={error}
        />
      )}
    </div>
  );
};

export default App;