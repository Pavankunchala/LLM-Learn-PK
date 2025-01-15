# CrewAI Documentation Assistant

An AI-powered assistant designed to answer queries about **CrewAI documentation** in real-time using advanced retrieval-augmented generation (RAG) techniques. The assistant leverages state-of-the-art models and vector search technologies for fast, accurate, and context-aware responses.

## 🚀 Features

- **Real-Time Streaming Responses:** Provides smooth, live answer generation for a better user experience.
- **Context-Aware Q&A:** Answers are generated strictly from the CrewAI documentation vector store.
- **Fallback Responses:** If the answer isn't found in the knowledge base, the assistant informs the user.
- **Step-by-Step Explanations:** Offers detailed and clear explanations when needed.
- **Automated Web Crawling:** Uses **Crawl4AI** to crawl and fetch CrewAI documentation dynamically.

## 🛠️ Tech Stack

- **LangChain:** For building the retrieval and Q&A pipeline.
- **Ollama (phi4 model):** For scalable and efficient LLM deployment.
- **Chroma Vector Store:** For efficient document storage and retrieval.
- **Chainlit:** For interactive and responsive UI with streaming.
- **Crawl4AI:** For automated crawling of CrewAI documentation to build the vector store.



## 📦 Installation

0 . **Crawl4AI Setup:** 

```bash
# Install the package
pip install -U crawl4ai

# Run post-installation setup
crawl4ai-setup

# Verify your installation
crawl4ai-doctor
```

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/Pavankunchala/LLM-Learn-PK/tree/main/Chainlit-apps/crewai-documentation-bot
   cd crewai-documentation-bot
   ```

2. **Create and Activate Virtual Environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Web Crawler to Populate the Vector Store:**
   ```bash
   python crawler.py  # Make sure Crawl4AI is properly configured
   ```

5. **Run the Application:**
   ```bash
   chainlit run main.py 
   ```
## 📖 How It Works

1. **Document Crawling:** Uses **Crawl4AI** to scrape the latest CrewAI documentation.
2. **Document Embedding:** Embedded into a Chroma vector store using HuggingFace embeddings.
3. **Query Handling:** User queries are processed and searched against the vector store.
4. **Answer Generation:** Responses are generated using the Ollama `phi4` model and streamed live.
5. **Interactive UI:** Chainlit provides a seamless chat interface for user interaction.

## 📝 Usage

1. Launch the app using the instructions above.
2. Type your question in the chat interface (e.g., "How does CrewAI handle workflows?").
3. The assistant will stream a step-by-step, accurate answer based on CrewAI documentation.

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests for improvements.

## 📬 Connect with Me

If you're working on exciting AI projects or looking for innovative talent, let's connect and explore potential collaborations!, you can connect with me on pavankunchalapkofficial@gmail.com

