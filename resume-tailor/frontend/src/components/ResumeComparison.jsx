import React, { useState, useEffect } from 'react';
import { extractLinks, validateResumePreservation } from '../utils/resumeParser';

const ResumeComparison = ({ originalLatex, tailoredLatex }) => {
  const [comparisonData, setComparisonData] = useState(null);
  const [view, setView] = useState('side-by-side');
  const [activeTab, setActiveTab] = useState('structure');

  useEffect(() => {
    if (originalLatex && tailoredLatex) {
      // Parse and analyze both versions
      const validation = validateResumePreservation(originalLatex, tailoredLatex);
      const originalLinks = extractLinks(originalLatex);
      const tailoredLinks = extractLinks(tailoredLatex);
      
      // Identify keywords that were added in the tailored version
      const originalWords = new Set(originalLatex.split(/\s+/).map(w => w.toLowerCase()));
      const tailoredWords = tailoredLatex.split(/\s+/);
      const newWords = tailoredWords.filter(word => {
        const normalizedWord = word.toLowerCase();
        return normalizedWord.length > 4 && !originalWords.has(normalizedWord);
      });
      
      setComparisonData({
        validation,
        links: {
          original: originalLinks,
          tailored: tailoredLinks,
          preserved: tailoredLinks.length >= originalLinks.length
        },
        newKeywords: [...new Set(newWords)].slice(0, 20),
        metrics: {
          originalLength: originalLatex.length,
          tailoredLength: tailoredLatex.length,
          lengthChange: ((tailoredLatex.length - originalLatex.length) / originalLatex.length * 100).toFixed(1)
        }
      });
    }
  }, [originalLatex, tailoredLatex]);

  if (!comparisonData) {
    return <div className="loading">Analyzing resume versions...</div>;
  }

  const { validation, links, newKeywords, metrics } = comparisonData;

  return (
    <div className="resume-comparison">
      <div className="comparison-header">
        <h3 className="comparison-title">Resume Comparison</h3>
        <div className="view-toggle">
          <button 
            className={`view-button ${view === 'side-by-side' ? 'active' : ''}`}
            onClick={() => setView('side-by-side')}
          >
            <span className="view-icon">⊞</span>
            Side by Side
          </button>
          <button 
            className={`view-button ${view === 'changes' ? 'active' : ''}`}
            onClick={() => setView('changes')}
          >
            <span className="view-icon">↭</span>
            Changes Only
          </button>
        </div>
      </div>
      
      <div className="comparison-tabs">
        <button 
          className={`tab-button ${activeTab === 'structure' ? 'active' : ''}`}
          onClick={() => setActiveTab('structure')}
        >
          <span className="tab-icon">🏗️</span>
          Structure
        </button>
        <button 
          className={`tab-button ${activeTab === 'links' ? 'active' : ''}`}
          onClick={() => setActiveTab('links')}
        >
          <span className="tab-icon">🔗</span>
          Links
        </button>
        <button 
          className={`tab-button ${activeTab === 'keywords' ? 'active' : ''}`}
          onClick={() => setActiveTab('keywords')}
        >
          <span className="tab-icon">🔍</span>
          Keywords
        </button>
        <button 
          className={`tab-button ${activeTab === 'metrics' ? 'active' : ''}`}
          onClick={() => setActiveTab('metrics')}
        >
          <span className="tab-icon">📊</span>
          Metrics
        </button>
      </div>
      
      <div className="comparison-content">
        {activeTab === 'structure' && (
          <div className="structure-comparison">
            <div className="comparison-status">
              <div className={`status-indicator ${validation.valid ? 'success' : 'warning'}`}>
                <span className="status-icon">{validation.valid ? '✓' : '⚠️'}</span>
                <span className="status-text">
                  {validation.valid 
                    ? 'All sections and structure preserved successfully' 
                    : 'Some structural elements may not be fully preserved'}
                </span>
              </div>
            </div>
            
            {!validation.valid && validation.issues.length > 0 && (
              <div className="issues-list">
                <h4 className="issues-title">Issues Detected:</h4>
                <ul>
                  {validation.issues.map((issue, index) => (
                    <li key={index} className="issue-item">{issue}</li>
                  ))}
                </ul>
              </div>
            )}
            
            {view === 'side-by-side' ? (
              <div className="side-by-side-view">
                <div className="original-column">
                  <h4 className="column-title">Original Resume</h4>
                  <div className="metadata-table">
                    <div className="metadata-row">
                      <span className="metadata-label">Sections:</span>
                      <span className="metadata-value">{validation.originalMetadata?.sectionCount || 0}</span>
                    </div>
                    <div className="metadata-row">
                      <span className="metadata-label">Companies:</span>
                      <span className="metadata-value">{validation.originalMetadata?.companyCount || 0}</span>
                    </div>
                    <div className="metadata-row">
                      <span className="metadata-label">Links:</span>
                      <span className="metadata-value">{validation.originalMetadata?.linkCount || 0}</span>
                    </div>
                  </div>
                </div>
                <div className="comparison-divider">→</div>
                <div className="tailored-column">
                  <h4 className="column-title">Tailored Resume</h4>
                  <div className="metadata-table">
                    <div className="metadata-row">
                      <span className="metadata-label">Sections:</span>
                      <span className="metadata-value">{validation.tailoredMetadata?.sectionCount || 0}</span>
                    </div>
                    <div className="metadata-row">
                      <span className="metadata-label">Companies:</span>
                      <span className="metadata-value">{validation.tailoredMetadata?.companyCount || 0}</span>
                    </div>
                    <div className="metadata-row">
                      <span className="metadata-label">Links:</span>
                      <span className="metadata-value">{validation.tailoredMetadata?.linkCount || 0}</span>
                    </div>
                  </div>
                </div>
              </div>
            ) : (
              <div className="changes-view">
                <h4 className="changes-title">Structure Changes</h4>
                <div className="changes-summary">
                  <div className="change-item">
                    <span className="change-label">Sections:</span>
                    <span className="change-value">
                      {validation.tailoredMetadata?.sectionCount === validation.originalMetadata?.sectionCount
                        ? 'All sections preserved ✓'
                        : `${validation.tailoredMetadata?.sectionCount} / ${validation.originalMetadata?.sectionCount} sections preserved`}
                    </span>
                  </div>
                  <div className="change-item">
                    <span className="change-label">Companies:</span>
                    <span className="change-value">
                      {validation.tailoredMetadata?.companyCount === validation.originalMetadata?.companyCount
                        ? 'All companies preserved ✓'
                        : `${validation.tailoredMetadata?.companyCount} / ${validation.originalMetadata?.companyCount} companies preserved`}
                    </span>
                  </div>
                  <div className="change-item">
                    <span className="change-label">Links:</span>
                    <span className="change-value">
                      {validation.tailoredMetadata?.linkCount >= validation.originalMetadata?.linkCount
                        ? 'All links preserved ✓'
                        : `${validation.tailoredMetadata?.linkCount} / ${validation.originalMetadata?.linkCount} links preserved`}
                    </span>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
        
        {activeTab === 'links' && (
          <div className="links-comparison">
            <div className="comparison-status">
              <div className={`status-indicator ${links.preserved ? 'success' : 'warning'}`}>
                <span className="status-icon">{links.preserved ? '✓' : '⚠️'}</span>
                <span className="status-text">
                  {links.preserved 
                    ? 'All links preserved successfully' 
                    : 'Some links may not be preserved'}
                </span>
              </div>
            </div>
            
            {view === 'side-by-side' ? (
              <div className="side-by-side-view">
                <div className="original-column">
                  <h4 className="column-title">Original Links ({links.original.length})</h4>
                  {links.original.length > 0 ? (
                    <ul className="links-list">
                      {links.original.map((link, index) => (
                        <li key={index} className="link-item">
                          <span className="link-text">{link.text}</span>
                          <span className="link-url">{link.url}</span>
                        </li>
                      ))}
                    </ul>
                  ) : (
                    <p className="no-links">No links found in original resume</p>
                  )}
                </div>
                <div className="comparison-divider">→</div>
                <div className="tailored-column">
                  <h4 className="column-title">Tailored Links ({links.tailored.length})</h4>
                  {links.tailored.length > 0 ? (
                    <ul className="links-list">
                      {links.tailored.map((link, index) => (
                        <li key={index} className="link-item">
                          <span className="link-text">{link.text}</span>
                          <span className="link-url">{link.url}</span>
                        </li>
                      ))}
                    </ul>
                  ) : (
                    <p className="no-links">No links found in tailored resume</p>
                  )}
                </div>
              </div>
            ) : (
              <div className="changes-view">
                <h4 className="changes-title">Link Changes</h4>
                <div className="link-preservation">
                  <div className="preservation-status">
                    <div className="preservation-label">Link Preservation:</div>
                    <div className="preservation-value">
                      {links.tailored.length >= links.original.length
                        ? `${links.tailored.length}/${links.original.length} links (100%)`
                        : `${links.tailored.length}/${links.original.length} links (${Math.round(links.tailored.length / links.original.length * 100)}%)`}
                    </div>
                  </div>
                  
                  {links.original.length > links.tailored.length && (
                    <div className="missing-links">
                      <h5 className="missing-title">Missing Links:</h5>
                      <ul className="links-list">
                        {links.original
                          .filter(origLink => !links.tailored.some(tailLink => tailLink.url === origLink.url))
                          .map((link, index) => (
                            <li key={index} className="link-item missing">
                              <span className="link-text">{link.text}</span>
                              <span className="link-url">{link.url}</span>
                            </li>
                          ))}
                      </ul>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        )}
        
        {activeTab === 'keywords' && (
          <div className="keywords-comparison">
            <div className="keywords-intro">
              <h4 className="keywords-title">New Keywords in Tailored Resume</h4>
              <p className="keywords-description">
                These words were added to match the job description better
              </p>
            </div>
            
            <div className="keywords-cloud">
              {newKeywords.length > 0 ? (
                newKeywords.map((keyword, index) => (
                  <div key={index} className="keyword-badge">
                    {keyword}
                  </div>
                ))
              ) : (
                <p className="no-keywords">No significant new keywords detected</p>
              )}
            </div>
          </div>
        )}
        
        {activeTab === 'metrics' && (
          <div className="metrics-comparison">
            <div className="metrics-cards">
              <div className="metric-card">
                <h4 className="metric-title">Content Length</h4>
                <div className="metric-value">{metrics.lengthChange}%</div>
                <div className="metric-description">
                  {metrics.lengthChange > 0 
                    ? 'More detailed content'
                    : metrics.lengthChange < 0
                      ? 'More concise content'
                      : 'Same length content'}
                </div>
                <div className="metric-details">
                  Original: {metrics.originalLength} chars<br />
                  Tailored: {metrics.tailoredLength} chars
                </div>
              </div>
              
              <div className="metric-card">
                <h4 className="metric-title">Structure Preservation</h4>
                <div className="metric-value">
                  {validation.valid ? '100%' : `${Math.round(100 - (validation.issues.length * 10))}%`}
                </div>
                <div className="metric-description">
                  {validation.valid
                    ? 'Perfect structure preservation'
                    : 'Some structural issues detected'}
                </div>
                <div className="metric-details">
                  {validation.valid
                    ? 'All sections, companies, and links preserved'
                    : `${validation.issues.length} issue(s) detected`}
                </div>
              </div>
              
              <div className="metric-card">
                <h4 className="metric-title">Link Preservation</h4>
                <div className="metric-value">
                  {links.original.length === 0
                    ? 'N/A'
                    : `${Math.round(links.tailored.length / Math.max(links.original.length, 1) * 100)}%`}
                </div>
                <div className="metric-description">
                  {links.original.length === 0
                    ? 'No links in original resume'
                    : links.tailored.length >= links.original.length
                      ? 'All links preserved'
                      : `${links.tailored.length}/${links.original.length} links preserved`}
                </div>
                <div className="metric-details">
                  Original: {links.original.length} links<br />
                  Tailored: {links.tailored.length} links
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      <style jsx>{`
        .resume-comparison {
          background-color: white;
          border-radius: 0.75rem;
          box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
          overflow: hidden;
          margin-bottom: 2rem;
        }
        
        .comparison-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 1.25rem 1.5rem;
          background-color: var(--gray-50);
          border-bottom: 1px solid var(--gray-200);
        }
        
        .comparison-title {
          font-size: 1.25rem;
          color: var(--gray-900);
          margin: 0;
        }
        
        .view-toggle {
          display: flex;
          gap: 0.5rem;
        }
        
        .view-button {
          display: flex;
          align-items: center;
          gap: 0.25rem;
          padding: 0.5rem 0.75rem;
          border-radius: 0.375rem;
          font-size: 0.875rem;
          border: 1px solid var(--gray-300);
          background-color: white;
          color: var(--gray-700);
          cursor: pointer;
          transition: all 0.2s;
        }
        
        .view-button:hover {
          background-color: var(--gray-100);
        }
        
        .view-button.active {
          background-color: var(--primary-lightest);
          border-color: var(--primary-light);
          color: var(--primary);
        }
        
        .view-icon {
          font-size: 1rem;
        }
        
        .comparison-tabs {
          display: flex;
          padding: 0 1.5rem;
          border-bottom: 1px solid var(--gray-200);
          overflow-x: auto;
        }
        
        .tab-button {
          display: flex;
          align-items: center;
          gap: 0.375rem;
          padding: 1rem 1.25rem;
          color: var(--gray-600);
          background: none;
          border: none;
          font-size: 0.95rem;
          font-weight: 500;
          border-bottom: 2px solid transparent;
          cursor: pointer;
          transition: all 0.2s;
          white-space: nowrap;
        }
        
        .tab-button:hover {
          color: var(--gray-900);
        }
        
        .tab-button.active {
          color: var(--primary);
          border-bottom-color: var(--primary);
        }
        
        .tab-icon {
          font-size: 1.1rem;
        }
        
        .comparison-content {
          padding: 1.5rem;
        }
        
        .comparison-status {
          margin-bottom: 1.5rem;
        }
        
        .status-indicator {
          display: flex;
          align-items: center;
          gap: 0.75rem;
          padding: 0.875rem;
          border-radius: 0.5rem;
        }
        
        .status-indicator.success {
          background-color: #ecfdf5;
          color: #065f46;
        }
        
        .status-indicator.warning {
          background-color: #fffbeb;
          color: #92400e;
        }
        
        .status-icon {
          font-size: 1.25rem;
        }
        
        .issues-list {
          background-color: #fff7ed;
          border-radius: 0.5rem;
          padding: 1rem 1.5rem;
          margin-bottom: 1.5rem;
        }
        
        .issues-title {
          color: #9a3412;
          margin-top: 0;
          margin-bottom: 0.75rem;
          font-size: 1rem;
        }
        
        .issue-item {
          color: #9a3412;
          margin-bottom: 0.375rem;
        }
        
        .side-by-side-view {
          display: flex;
          gap: 1.5rem;
        }
        
        .original-column, .tailored-column {
          flex: 1;
          min-width: 0;
        }
        
        .comparison-divider {
          display: flex;
          align-items: center;
          color: var(--gray-400);
          font-size: 1.5rem;
        }
        
        .column-title {
          margin-top: 0;
          margin-bottom: 1rem;
          color: var(--gray-700);
          font-size: 1.1rem;
          padding-bottom: 0.5rem;
          border-bottom: 1px solid var(--gray-200);
        }
        
        .metadata-table {
          display: flex;
          flex-direction: column;
          gap: 0.5rem;
        }
        
        .metadata-row {
          display: flex;
          justify-content: space-between;
          padding: 0.5rem 0;
          border-bottom: 1px solid var(--gray-100);
        }
        
        .metadata-label {
          color: var(--gray-600);
          font-weight: 500;
        }
        
        .metadata-value {
          color: var(--gray-900);
          font-weight: 600;
        }
        
        .changes-view {
          padding: 1rem;
          background-color: var(--gray-50);
          border-radius: 0.5rem;
        }
        
        .changes-title {
          margin-top: 0;
          margin-bottom: 1rem;
          color: var(--gray-800);
          font-size: 1.1rem;
        }
        
        .changes-summary {
          display: flex;
          flex-direction: column;
          gap: 0.75rem;
        }
        
        .change-item {
          display: flex;
          justify-content: space-between;
          padding: 0.75rem;
          background-color: white;
          border-radius: 0.375rem;
          box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        }
        
        .change-label {
          color: var(--gray-600);
          font-weight: 500;
        }
        
        .change-value {
          color: var(--gray-900);
          font-weight: 600;
        }
        
        .links-list {
          list-style: none;
          padding: 0;
          margin: 0;
          display: flex;
          flex-direction: column;
          gap: 0.5rem;
        }
        
        .link-item {
          display: flex;
          flex-direction: column;
          padding: 0.75rem;
          background-color: var(--gray-50);
          border-radius: 0.375rem;
          border-left: 3px solid var(--primary-light);
          font-size: 0.9rem;
        }
        
        .link-item.missing {
          border-left-color: #f59e0b;
          background-color: #fffbeb;
        }
        
        .link-text {
          font-weight: 600;
          color: var(--gray-800);
          margin-bottom: 0.25rem;
        }
        
        .link-url {
          color: var(--primary);
          word-break: break-all;
          font-size: 0.85rem;
        }
        
        .no-links {
          color: var(--gray-500);
          font-style: italic;
          text-align: center;
          padding: 1rem;
        }
        
        .keywords-cloud {
          display: flex;
          flex-wrap: wrap;
          gap: 0.75rem;
          margin-top: 1.5rem;
          padding: 1.5rem;
          background-color: var(--gray-50);
          border-radius: 0.5rem;
        }
        
        .keyword-badge {
          padding: 0.5rem 1rem;
          background-color: var(--primary-lightest);
          color: var(--primary-dark);
          border-radius: 2rem;
          font-size: 0.9rem;
          font-weight: 500;
          box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
        }
        
        .no-keywords {
          color: var(--gray-500);
          font-style: italic;
          text-align: center;
          padding: 1rem;
          width: 100%;
        }
        
        .metrics-cards {
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
          gap: 1.5rem;
        }
        
        .metric-card {
          background-color: white;
          border: 1px solid var(--gray-200);
          border-radius: 0.75rem;
          padding: 1.5rem;
          text-align: center;
          box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
          transition: transform 0.2s, box-shadow 0.2s;
        }
        
        .metric-card:hover {
          transform: translateY(-3px);
          box-shadow: 0 5px 15px rgba(0, 0, 0, 0.08);
        }
        
        .metric-title {
          color: var(--gray-600);
          font-size: 1rem;
          margin-top: 0;
          margin-bottom: 0.75rem;
        }
        
        .metric-value {
          font-size: 2rem;
          font-weight: 700;
          color: var(--gray-900);
          margin-bottom: 0.5rem;
        }
        
        .metric-description {
          color: var(--gray-600);
          margin-bottom: 1rem;
        }
        
        .metric-details {
          font-size: 0.85rem;
          color: var(--gray-500);
          padding-top: 0.75rem;
          border-top: 1px solid var(--gray-100);
        }
        
        @media (max-width: 768px) {
          .side-by-side-view {
            flex-direction: column;
            gap: 2rem;
          }
          
          .comparison-divider {
            transform: rotate(90deg);
            align-self: center;
          }
          
          .metrics-cards {
            grid-template-columns: 1fr;
          }
        }
      `}</style>
    </div>
  );
};

export default ResumeComparison;