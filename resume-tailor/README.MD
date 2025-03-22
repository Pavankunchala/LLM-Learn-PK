# Resume Tailor

A tool that uses AI to tailor your resume to specific job descriptions by leveraging LLMs through Ollama.

## Features

- Upload or paste your resume
- Enter a job description to tailor your resume for
- Select from different LLM models
- Generate a tailored resume in LaTeX format
- Compile LaTeX to PDF
- Download both LaTeX and PDF versions

## Setup Instructions

### Prerequisites

- Node.js (v16+)
- Python (v3.9+)
- LaTeX distribution (TeX Live, MiKTeX, etc.) installed on your system
- [Ollama](https://ollama.ai/) installed and running locally

### Frontend Setup

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm run dev
```

The frontend will be available at http://localhost:5173

### Backend Setup

1. Navigate to the backend directory:
```bash
cd backend
```

2. Create a virtual environment:
```bash
python -m venv venv
```

3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - macOS/Linux: `source venv/bin/activate`

4. Install dependencies:
```bash
pip install -r requirements.txt
```

5. Start the Flask server:
```bash
python app.py
```

The backend API will be available at http://localhost:5000

### Ollama Setup

1. Make sure Ollama is installed on your system. Visit [ollama.ai](https://ollama.ai/) for installation instructions.

2. Pull the required models:
```bash
ollama pull llama3
ollama pull mistral
# Add other models as needed
```

## Usage

1. Open your browser and go to http://localhost:5173
2. Upload or paste your resume
3. Enter the job description
4. Select the appropriate LLM model
5. Click "Generate Tailored Resume"
6. View and download your tailored resume in LaTeX or PDF format

## Known Limitations

- PDF compilation requires LaTeX to be installed on the system
- The quality of the tailored resume depends on the LLM model used
- Large resumes or job descriptions may take longer to process

## Troubleshooting

- If PDF compilation fails, you can still download the LaTeX file and compile it manually
- Make sure Ollama is running before starting the application
- Check that you have the required models pulled in Ollama

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.