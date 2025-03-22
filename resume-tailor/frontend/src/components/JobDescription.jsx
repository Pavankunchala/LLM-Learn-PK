import React from 'react';

const JobDescription = ({ value, onChange, onBack, onNext }) => {
  return (
    <div className="space-y-6">
      <div>
        <label htmlFor="job-description" className="block text-sm font-medium text-gray-700 mb-2">
          Job Description
        </label>
        <p className="text-sm text-gray-500 mb-2">
          Paste the job description for which you'd like to tailor your resume
        </p>
        <textarea
          id="job-description"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          rows={10}
          className="textarea-field"
          placeholder="Paste the job description here..."
        ></textarea>
      </div>
      
      <div className="flex justify-between">
        <button
          type="button"
          className="btn btn-secondary"
          onClick={onBack}
        >
          Back
        </button>
        
        <button
          type="button"
          className="btn btn-primary"
          onClick={onNext}
          disabled={!value.trim()}
        >
          Next
        </button>
      </div>
    </div>
  );
};

export default JobDescription;