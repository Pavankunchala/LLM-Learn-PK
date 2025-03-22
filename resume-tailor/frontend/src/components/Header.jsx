import React, { useState, useEffect } from 'react';

const Header = () => {
  const [isScrolled, setIsScrolled] = useState(false);
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

  // Handle scroll event to add shadow on scroll
  useEffect(() => {
    const handleScroll = () => {
      if (window.scrollY > 50) {
        setIsScrolled(true);
      } else {
        setIsScrolled(false);
      }
    };

    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  return (
    <header className={isScrolled ? 'scrolled' : ''}>
      <div className="container">
        <div className="logo">
          <div className="logo-icon">📄</div>
          <div className="logo-text">Resume Tailor</div>
        </div>
        
        {/* Desktop Navigation */}
        <nav className="nav-links">
          <a href="#" className="nav-link">Home</a>
          <a href="#" className="nav-link">How It Works</a>
          <a href="#" className="nav-link">Examples</a>
          <a href="#" className="nav-link">Pricing</a>
          <a href="https://github.com/your-username/resume-tailor" 
             target="_blank" 
             rel="noopener noreferrer"
             className="nav-link github-link">
            <svg className="github-icon" fill="currentColor" viewBox="0 0 24 24">
              <path fillRule="evenodd" d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z" clipRule="evenodd" />
            </svg>
            GitHub
          </a>
        </nav>
        
        {/* Mobile Menu Toggle */}
        <div className="mobile-menu-toggle" onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}>
          <svg xmlns="http://www.w3.org/2000/svg" className="mobile-menu-icon" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d={isMobileMenuOpen ? "M6 18L18 6M6 6l12 12" : "M4 6h16M4 12h16M4 18h16"} />
          </svg>
        </div>
        
        {/* Mobile Navigation */}
        <div className={`mobile-menu ${isMobileMenuOpen ? 'open' : ''}`}>
          <div className="mobile-nav-links">
            <a href="#" className="mobile-nav-link" onClick={() => setIsMobileMenuOpen(false)}>Home</a>
            <a href="#" className="mobile-nav-link" onClick={() => setIsMobileMenuOpen(false)}>How It Works</a>
            <a href="#" className="mobile-nav-link" onClick={() => setIsMobileMenuOpen(false)}>Examples</a>
            <a href="#" className="mobile-nav-link" onClick={() => setIsMobileMenuOpen(false)}>Pricing</a>
            <a href="https://github.com/your-username/resume-tailor" 
               target="_blank" 
               rel="noopener noreferrer"
               className="mobile-nav-link github-mobile-link"
               onClick={() => setIsMobileMenuOpen(false)}>
              <svg className="github-icon" fill="currentColor" viewBox="0 0 24 24">
                <path fillRule="evenodd" d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z" clipRule="evenodd" />
              </svg>
              GitHub
            </a>
          </div>
        </div>
        
        {/* Add mobile menu styles */}
        <style jsx>{`
          .mobile-menu-toggle {
            display: none;
            cursor: pointer;
          }
          
          .mobile-menu-icon {
            width: 28px;
            height: 28px;
            color: var(--gray-700);
          }
          
          .mobile-menu {
            display: none;
            position: fixed;
            top: 70px;
            left: 0;
            right: 0;
            background: white;
            padding: 1rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            z-index: 99;
            transform: translateY(-100%);
            opacity: 0;
            transition: all 0.3s ease;
          }
          
          .mobile-menu.open {
            transform: translateY(0);
            opacity: 1;
          }
          
          .mobile-nav-links {
            display: flex;
            flex-direction: column;
            gap: 1rem;
          }
          
          .mobile-nav-link {
            color: var(--gray-700);
            text-decoration: none;
            font-weight: 500;
            padding: 0.75rem;
            border-radius: 0.5rem;
            transition: all 0.2s ease;
          }
          
          .mobile-nav-link:hover {
            background-color: var(--gray-100);
            color: var(--primary);
          }
          
          .github-mobile-link {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            margin-top: 0.5rem;
            padding: 0.75rem;
            background-color: var(--gray-100);
            border-radius: 0.5rem;
          }
          
          @media (max-width: 768px) {
            .mobile-menu-toggle {
              display: block;
            }
            
            .mobile-menu {
              display: block;
            }
            
            .nav-links {
              display: none;
            }
          }
          
          /* Fix for header nav links */
          .nav-links {
            display: flex;
            flex-direction: row;
            align-items: center;
          }
          
          .nav-link {
            display: inline-flex;
            align-items: center;
          }
        `}</style>
      </div>
    </header>
  );
};

export default Header;