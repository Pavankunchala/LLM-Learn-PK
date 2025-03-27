import React from 'react';

const TemplateSelection = ({ selectedTemplate, onSelect }) => {
  const templates = [
    { 
      id: 'classic', 
      name: 'Classic', 
      description: 'Traditional resume layout with clean formatting', 
      icon: '📄',
      preview: '/templates/classic-preview.png'
    },
    { 
      id: 'modern', 
      name: 'Modern', 
      description: 'Contemporary design with stylish elements', 
      icon: '🎨',
      preview: '/templates/modern-preview.png'
    },
    { 
      id: 'technical', 
      name: 'Technical', 
      description: 'Optimized for technical roles with skills emphasis', 
      icon: '💻',
      preview: '/templates/technical-preview.png'
    },
    { 
      id: 'academic', 
      name: 'Academic', 
      description: 'Focused on educational background and publications', 
      icon: '🎓',
      preview: '/templates/academic-preview.png'
    },
  ];

  return (
    <div className="template-selection">
      <h2 className="section-title">Choose a Resume Template</h2>
      <p className="section-description">
        Select a template that best represents your professional style
      </p>

      <div className="templates-grid">
        {templates.map((template) => (
          <div 
            key={template.id}
            className={`template-card ${selectedTemplate === template.id ? 'selected' : ''}`}
            onClick={() => onSelect(template.id)}
          >
            <div className="template-icon">{template.icon}</div>
            <div className="template-preview">
              <img 
                src={template.preview} 
                alt={`${template.name} template preview`}
                onError={(e) => {
                  e.target.onerror = null;
                  e.target.src = 'https://via.placeholder.com/200x280?text=Preview+Not+Available';
                }}
              />
            </div>
            <div className="template-info">
              <h3 className="template-name">{template.name}</h3>
              <p className="template-description">{template.description}</p>
            </div>
            {selectedTemplate === template.id && (
              <div className="selected-badge">Selected</div>
            )}
          </div>
        ))}
      </div>

      <style jsx>{`
        .template-selection {
          margin-bottom: 2rem;
        }
        
        .section-title {
          text-align: center;
          margin-bottom: 0.5rem;
        }
        
        .section-description {
          text-align: center;
          color: var(--gray-600);
          margin-bottom: 2rem;
        }
        
        .templates-grid {
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
          gap: 1.5rem;
        }
        
        .template-card {
          border: 2px solid var(--gray-200);
          border-radius: 0.75rem;
          overflow: hidden;
          cursor: pointer;
          transition: all 0.3s ease;
          position: relative;
          background-color: white;
        }
        
        .template-card:hover {
          transform: translateY(-5px);
          box-shadow: 0 8px 15px rgba(0, 0, 0, 0.1);
          border-color: var(--gray-300);
        }
        
        .template-card.selected {
          border-color: var(--primary);
          box-shadow: 0 8px 20px rgba(59, 130, 246, 0.25);
          transform: translateY(-5px);
        }
        
        .template-icon {
          position: absolute;
          top: 0.75rem;
          left: 0.75rem;
          font-size: 1.5rem;
          z-index: 1;
          background-color: white;
          width: 2.5rem;
          height: 2.5rem;
          display: flex;
          align-items: center;
          justify-content: center;
          border-radius: 0.5rem;
          box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
        }
        
        .template-preview {
          height: 280px;
          overflow: hidden;
          display: flex;
          align-items: center;
          justify-content: center;
          background-color: var(--gray-50);
          border-bottom: 1px solid var(--gray-200);
        }
        
        .template-preview img {
          max-width: 100%;
          max-height: 100%;
          object-fit: contain;
        }
        
        .template-info {
          padding: 1rem;
        }
        
        .template-name {
          font-size: 1.1rem;
          margin-bottom: 0.5rem;
          color: var(--gray-900);
        }
        
        .template-description {
          font-size: 0.9rem;
          color: var(--gray-600);
          line-height: 1.4;
        }
        
        .selected-badge {
          position: absolute;
          top: 0.75rem;
          right: 0.75rem;
          background-color: var(--primary);
          color: white;
          font-size: 0.8rem;
          font-weight: 500;
          padding: 0.25rem 0.5rem;
          border-radius: 0.25rem;
          box-shadow: 0 2px 5px rgba(59, 130, 246, 0.3);
        }
        
        @media (max-width: 640px) {
          .templates-grid {
            grid-template-columns: 1fr;
          }
        }
      `}</style>
    </div>
  );
};

export default TemplateSelection;