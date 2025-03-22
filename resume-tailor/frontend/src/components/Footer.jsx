import React from 'react';

const Footer = () => {
  return (
    <footer>
      <div className="container">
        <div className="footer-content">
          <div className="footer-info">
            <h3 className="footer-title">Resume Tailor</h3>
            <p className="footer-description">
              Create tailored resumes for specific job descriptions using the power of AI.
              Increase your chances of getting interviews by highlighting relevant skills and experiences.
            </p>
          </div>
          
          <div className="footer-navigation">
            <h4 className="footer-nav-title">Quick Links</h4>
            <ul className="footer-nav-list">
              <li><a href="#" className="footer-link">Home</a></li>
              <li><a href="#" className="footer-link">How It Works</a></li>
              <li><a href="#" className="footer-link">About</a></li>
              <li><a href="#" className="footer-link">GitHub</a></li>
            </ul>
          </div>
          
          <div className="footer-legal">
            <h4 className="footer-legal-title">Legal</h4>
            <ul className="footer-legal-list">
              <li><a href="#" className="footer-link">Privacy Policy</a></li>
              <li><a href="#" className="footer-link">Terms of Service</a></li>
            </ul>
          </div>
        </div>
        
        <div className="footer-bottom">
          <p>&copy; {new Date().getFullYear()} Resume Tailor. All rights reserved.</p>
        </div>
      </div>
    </footer>
  );
};

export default Footer;