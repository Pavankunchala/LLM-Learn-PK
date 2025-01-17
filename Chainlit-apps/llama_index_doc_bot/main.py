import chainlit as cl
from langchain.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.schema import Document
import os

# === Step 1: Load the Chroma Vector Store ===
VECTOR_STORE_DIR = "./vectorstore"
embedding_model = HuggingFaceEmbeddings(model_name="dunzhang/stella_en_1.5B_v5")
vector_db = Chroma(persist_directory=VECTOR_STORE_DIR, embedding_function=embedding_model)

# === Step 2: Define the System Prompt ===
system_prompt = """
You are an expert AI assistant specialized in answering technical questions about the LLama-Index documentation.

Strict Source Usage:
- Provide answers strictly based on information retrieved from the LLama Index documentation stored in the vector database.
- If the answer cannot be found in the provided content, clearly state:
  "I could not find this information in the LLama Index documentation."

Step-by-Step Reasoning (Chain-of-Thought):
- Break down complex answers into clear, logical steps.
- Explain your reasoning step-by-step before presenting the final answer.

Mandatory Code Examples:
- Always include clear and well-commented code examples, even if the user doesn't explicitly ask.
- Use Python code blocks with proper formatting for readability.

Reference Sources:
- At the end of every answer, provide a "References" section with links or source titles from the vector store.
- In the main answer itself add markdown links as clicakble buttons to refer to that particular section



Polite and Helpful Communication:
- If a question is unrelated to LLama Index, respond politely:
  "I am designed to answer questions specifically about LLama Index documentation. Please ask about LLama Index to get accurate information."


"""

# === Step 3: Build the QA Pipeline ===
def build_qa_pipeline():
    llm = ChatOllama(model="qwen2.5:14b", base_url="http://localhost:11434", streaming=True)

    # Update prompt for better clarity
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{context}\n\nQuestion: {question}\nAnswer:")
    ])

    # MultiQueryRetriever for diverse document retrieval
    retriever = MultiQueryRetriever.from_llm(
        retriever=vector_db.as_retriever(search_kwargs={"k": 10}),
        llm=llm
    )

     # Function to flatten document content into plain text
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Create a pipeline with formatted document context
    qa_pipeline = (
    retriever
    | (lambda docs: {"context": format_docs(docs), "question": cl.user_session.get("current_query")})
    | prompt_template
    | llm
)

    return qa_pipeline

qa_pipeline = build_qa_pipeline()

# === Step 4: Chainlit UI Integration ===
@cl.on_chat_start
async def start_chat():
    cl.user_session.set("qa_pipeline", qa_pipeline)
    await cl.Message(
        content="📖 **Welcome to the LLama Index Documentation Assistant!**\n\nAsk me anything about LLama Index Documentation."
    ).send()

@cl.on_message
async def handle_message(message: cl.Message):
    qa_pipeline = cl.user_session.get("qa_pipeline")
    query = message.content

    cl.user_session.set("current_query", query)

    if query.lower() in ["exit", "quit"]:
        await cl.Message("Session ended. Come back anytime!").send()
        return

    # Initialize the streaming message
    msg = cl.Message(content="")

    # Stream the response with proper handling of AIMessageChunk
    async for chunk in qa_pipeline.astream({"question": query}):
        # Check if the chunk is an AIMessageChunk and extract the text
        if hasattr(chunk, "content"):
            token = chunk.content  # Extract content if it's an AIMessageChunk
        else:
            token = str(chunk)  # Fallback in case it's a plain string

        await msg.stream_token(token)

    await msg.send()

if __name__ == "__main__":
    cl.run(port=8000, host="0.0.0.0")
