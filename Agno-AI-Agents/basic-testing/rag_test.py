from pathlib import Path
from agno.agent import Agent
from agno.models.ollama import Ollama
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.knowledge.pdf_url import PDFUrlKnowledgeBase
from agno.vectordb.chroma import ChromaDb
from agno.embedder.huggingface import HuggingfaceCustomEmbedder

# Instead of using pdf_path.as_uri(), use the URL from the local HTTP server.
pdf_path = "D:/LLM-Learn-PK/Agno-AI-Agents/basic-testing/new_resume.pdf"


# Initialize the Hugging Face embedder
embedder = HuggingfaceCustomEmbedder()

# Initialize ChromaDB
vector_db = ChromaDb(collection="test", path="chromadb", persistent_client=True, embedder = embedder)

# Create the knowledge base using the HTTP URL of the local PDF
knowledge_base = PDFUrlKnowledgeBase(
    document_lists = [pdf_path],
    vector_db=vector_db,
    embedder=embedder
)

# Load the knowledge base (only needed for the first run)
knowledge_base.load(recreate=False)

# Create the agent
agent = Agent(
    model=Ollama(id="llama3.2"),
    description="You are Resume Drafting Expert",
    instructions=[
        "You are an expert in resume drafting based on the attached document.",
        "If the question is better suited for the web, search the web to fill in gaps.",
        "Prefer the information in your knowledge base over web results."
    ],
    knowledge=knowledge_base,
    tools=[DuckDuckGoTools()],
    show_tool_calls=True,
    
)

# Generate and print the resume rewriting response for a Computer Vision Engineer role
agent.print_response("Rewrite my resume for a Computer Vision Engineer role", markdown=True)
