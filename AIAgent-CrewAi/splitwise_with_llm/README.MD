# Bill Splitting Automation with AI

This project simplifies the process of splitting bills by leveraging AI to analyze a PDF bill, interpret purchases, and generate a detailed breakdown of costs per person. It uses **CrewAI** and **VisionParser** to automate the process.

---

## Features

- 📄 **PDF Parsing**: Converts bill PDFs into Markdown format using VisionParser.
- 🤖 **AI-Powered Analysis**: Uses a CrewAI agent with a language model (`ollama/mistral-nemo`) to calculate and explain costs.
- 📊 **Detailed Outputs**:
  - Individual costs.
  - Shared costs for common items.
  - Tax calculations.
- 📝 **Markdown Report**: Outputs a detailed bill breakdown in a Markdown file.

---

## Prerequisites

1. **Python 3.8+**.
2. Install the required packages:
   ```
   pip install crewai vision-parse
```