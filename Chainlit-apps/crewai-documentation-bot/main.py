import chainlit as cl
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_ollama.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
import os

# === Step 1: Load the Chroma Vector Store ===
VECTOR_STORE_DIR = "./vectorstore"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vector_db = Chroma(persist_directory=VECTOR_STORE_DIR, embedding_function=embedding_model)

# === Step 2: Define the System Prompt ===
system_prompt = """
You are an intelligent AI assistant specialized in answering questions about the CrewAI  documentation.
Provide accurate, detailed, and clear answers using the CrewAIU docs.
If a question is unrelated to CrewAI , politely inform the user.
Offer step-by-step explanations and examples where helpful. Make sure your answers are always from the vector store information , if not make sure let the user know you  dont know the answer
You are an expert in CrewAI and you are intelligent enough to judge your own answers before answering 
Also add reference links in the end of the content
"""

# === Step 3: Build the QA Pipeline ===
def build_qa_pipeline():
    llm = ChatOllama(model="phi4", base_url="http://localhost:11434") # Using Ollama's phi4 model
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template=f"""
        {system_prompt}

        Context:
        {{context}}

        Question:
        {{question}}

        Answer:
        """
    )
    retriever = vector_db.as_retriever(search_kwargs={"k": 10})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt_template}
    )
    return qa_chain

qa_pipeline = build_qa_pipeline()

# === Step 4: Chainlit UI Integration ===
@cl.on_chat_start
async def start_chat():
    cl.user_session.set("qa_pipeline", qa_pipeline)
    await cl.Message(
        content="📖 **Welcome to the CrewAI Documentation Assistant!**\n\nAsk me anything about CrewAI ."
    ).send()

@cl.on_message
async def handle_message(message: cl.Message):
    qa_pipeline = cl.user_session.get("qa_pipeline")
    query = message.content
    

    if query.lower() in ["exit", "quit"]:
        await cl.Message("Session ended. Come back anytime!").send()
        return

    # Generate the response using the QA pipeline
    response = qa_pipeline.run(query)
    await cl.Message(content=f"**Answer:**\n\n{response}").send()

if __name__ == "__main__":
    cl.run(port=8000, host="0.0.0.0")
