from pathlib import Path
from agno.agent import Agent
from agno.models.ollama import Ollama
from agno.knowledge.pdf import PDFKnowledgeBase, PDFReader
from agno.vectordb.chroma import ChromaDb
from agno.embedder.ollama import OllamaEmbedder
from agno.tools.crawl4ai import Crawl4aiTools

# 1. Embedder & Vector DB setup
embedder = OllamaEmbedder(id="nomic-embed-text", dimensions=768)
vector_db = ChromaDb(
    collection="resume_kb",
    path="chromadb",
    persistent_client=True,
    embedder=embedder
)

# 2. Knowledge base: your local PDF resume
pdf_path = "D:/LLM-Learn-PK/Agno-AI-Agents/basic-testing/new_resume.pdf"
knowledge_base = PDFKnowledgeBase(
    path=pdf_path,
    vector_db=vector_db,
    embedder=embedder,
    reader=PDFReader(chunk=True, chunk_size=512),
)
knowledge_base.load(recreate=False)  # load existing index

# 3. Agent: only use KB for education & experience; use Crawl4aiTools for projects
agent = Agent(
    model=Ollama(id="llama3.2"),
    description="You are a Resume Drafting Expert.",
    instructions=[
        # System role: restrict to KB + Crawl4ai
        "You may only retrieve **education** and **experience** entries from the provided PDF knowledge base.",
        "For **projects**, use only the Crawl4ai tool to crawl the GitHub repository at "
        "`https://github.com/Pavankunchala/LLM-Learn-PK` and extract relevant project names and descriptions.",
        "Do NOT call any other tools or web APIs.",
        "",
        # Task instructions
        "Given the job description, rewrite the resume to:",
        "1. **Highlight** and **reorder** education & experience bullets from your KB to match the role.",
        "2. **Append** up to 5 GitHub projects (name + short summary) fetched by crawling the repo link.",
        "3. Preserve all original resume section headings and dates exactly as in the PDF.",
        "4. Do NOT invent or omit any KB content; only rephrase to align with the job requirements."
    ],
    knowledge=knowledge_base,
    tools=[Crawl4aiTools(max_length=None)],
    show_tool_calls=True,
)

# 4. Run!
job_description = """About the job
Neptune Technologies is a leading provider of innovative technology solutions, specializing in artificial intelligence and blockchain applications. We develop cutting-edge tools and platforms that empower organizations across multiple industries to make better decisions and optimize their operations. Our mission is to bring transformative AI technologies to various sectors, creating intelligent systems that grow alongside our clients. We are pioneers in developing adaptive tools that scale with our clients' needs and enhance their capabilities. Join us as we explore the future of technology-driven innovation, where accuracy meets advancement and AI creates new possibilities for businesses worldwide.


Role Description

We are seeking a Senior Generative AI Engineer to join our collaborative team and contribute to the development of sophisticated AI solutions across multiple industries. In this role, you will work closely with cross-functional teams to develop advanced generative AI systems and autonomous intelligent agents. Your expertise in generative models, deep learning frameworks, and agent architecture will be valuable in creating intelligent, scalable AI applications that can reason, plan, and execute tasks with minimal human intervention. We value team players who are proactive, independent, and resourceful in their approach to problem-solving. This position will begin as a hybrid role, with an expectation to transition to full-time in-person work in the future.


Core Qualifications:

Master's degree or Ph D with equivalent experience in Computer Science, Computer Engineering, Electrical Engineering, or related technical field
Experience working with both classical and modern machine learning models
Solid experience as a Python Developer
Experience developing LLM-based applications in Python, working with models such as OpenAI, Anthropic, xAI, and Google Gemini
Knowledge of advanced NLP techniques, including semantic analysis and language understanding
Experience with Retrieval-Augmented Generation (RAG) systems and vector databases such as Weaviate, PineCone, Chroma, and FAISS
Experience contributing to AI projects from conception to production deployment
Knowledge of prompt engineering, parameter-efficient fine-tuning, and building AI assistants
Familiarity with cutting-edge LLM techniques, including zero-shot learning, few-shot learning, and model fine-tuning
Experience with agent development and relevant frameworks, including OpenAI Agent SDK, AutoGPT, LangGraph, and other agentic systems
Experience with Generative AI Python frameworks, including Langchain, LlamaIndex, HuggingFace, and custom deployment frameworks
Familiarity with using LLMs for code development and experience with AI-based code development platforms such as Cursor AI, GitHub Copilot, and similar tools
Proficiency in CI/CD pipelines for AI systems and Git-based collaborative development workflows
Experience in optimizing and scaling AI models for production environments
Strong teamwork skills with the ability to collaborate effectively across departments
Proactive approach to identifying and solving technical challenges before they become problems
Independent work ethic with the ability to manage tasks with minimal supervision
Resourcefulness in finding creative solutions when faced with limited resources or information
Flexibility to work with some overlap in US Eastern Time (ET) hours for meetings and team collaboration


Preferred Qualifications:

Experience with open-source LLMs such as Llama, Mistral, DeepSeek, Qwen, etc.
Experience with real-time LLM inference optimization techniques (quantization, distillation, etc.)
Experience developing multimodal AI applications (text + image/audio/video)
Publications in top-tier AI/ML conferences such as NeurIPS, ICML, ACL, EMNLP, ICLR, or CVPR, particularly in generative AI topics
Experience working with financial and stock market data
Knowledge of financial markets and investment principles
Active contribution to the generative AI research community through publications, open-source projects, or technical blogs
Experience in collaborative AI engineering teams


Key Responsibilities:

Contribute to the architecture and development of our Enterprise AI platform, enabling sophisticated applications of Large-Language Models (LLMs) and Foundation Models (FMs)
Collaborate with team members to implement the technical vision and roadmap for AI initiatives
Help design and implement scalable API systems for accessing, fine-tuning, and deploying state-of-the-art LLMs
Develop and maintain advanced RAG pipelines and knowledge systems optimized for financial data
Design and develop AI agents capable of complex reasoning, tool use, and task execution for financial applications
Contribute to application-specific interfaces that leverage LLMs to enhance customer and associate experiences
Leverage AI-assisted coding tools to improve development efficiency and quality
Participate in establishing best practices for AI safety, security, and ethical guidelines
Work proactively and independently to solve complex technical challenges
Share knowledge and collaborate with team members to elevate the entire team's capabilities
Identify opportunities for innovation by evaluating emerging AI technologies




"""
agent.print_response(job_description, markdown=True)
