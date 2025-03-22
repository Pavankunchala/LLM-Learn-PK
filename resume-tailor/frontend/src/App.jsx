import { useState } from 'react';
import Header from './components/Header';
import Footer from './components/Footer';
import ResumeUpload from './components/ResumeUpload';
import JobDescription from './components/JobDescription';
import ModelSelection from './components/ModelSelection';
import ResultPreview from './components/ResultPreview';
import ProcessingOverlay from './components/ProcessingOverlay';

function App() {
  const [resumeData, setResumeData] = useState('');
  const [resumePdfBase64, setResumePdfBase64] = useState(null);
  const [jobDescription, setJobDescription] = useState('');
  const [selectedModel, setSelectedModel] = useState('llama3');
  const [isProcessing, setIsProcessing] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [activeStep, setActiveStep] = useState(1);

  const handleResumeUpload = (text, pdfBase64) => {
    setResumeData(text);
    setResumePdfBase64(pdfBase64);
    setActiveStep(2);
  };

  const handleJobDescriptionChange = (text) => {
    setJobDescription(text);
  };

  const handleModelChange = (model) => {
    setSelectedModel(model);
  };

  const processResume = async () => {
    if (!resumeData || !jobDescription) {
      setError('Please provide both a resume and job description');
      return;
    }

    setIsProcessing(true);
    setError(null);

    try {
      const requestBody = {
        resume: resumePdfBase64 || resumeData,
        resumeFormat: resumePdfBase64 ? 'pdf_base64' : 'text',
        jobDescription,
        model: selectedModel,
      };
      
      const response = await fetch('http://localhost:5000/process', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestBody),
      });

      if (!response.ok) throw new Error('Failed to process resume');

      const data = await response.json();
      if (data.error) throw new Error(data.error);
      
      setResult(data);
      setActiveStep(4);
    } catch (err) {
      setError(err.message || 'An error occurred while processing your resume');
    } finally {
      setIsProcessing(false);
    }
  };

  const resetApp = () => {
    setResumeData('');
    setResumePdfBase64(null);
    setJobDescription('');
    setSelectedModel('llama3');
    setResult(null);
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
                <ResumeUpload onUpload={handleResumeUpload} resumeData={resumeData} />
              )}
              
              {activeStep === 2 && (
                <JobDescription 
                  value={jobDescription} 
                  onChange={handleJobDescriptionChange} 
                  onBack={() => setActiveStep(1)}
                  onNext={() => setActiveStep(3)}
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
      <ProcessingOverlay isProcessing={isProcessing} />
    </div>
  );
}

export default App;
