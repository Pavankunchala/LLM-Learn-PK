import os
import base64
import tempfile
import subprocess
import re
import io
import logging
import time
import json
import hashlib
import functools
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Set, Union
from dataclasses import dataclass
from contextlib import contextmanager
from datetime import datetime

# Flask and CORS
from flask import Flask, request, jsonify
from flask_cors import CORS

# External LLM API library
import ollama

# Document processing
import PyPDF2
from io import BytesIO

# LangChain imports - using updated import paths to avoid deprecation warnings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter

# Configure logging with structured format and both file and console output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("resume_tailor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("resume_tailor")

# ------------------ Configuration ------------------
class Config:
    """Configuration class for the application."""
    DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "llama3")  # Can be overridden with environment variable
    USE_GPU = os.getenv("USE_GPU", "1") == "1"  # Use GPU by default if available
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "2000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "400"))
    BM25_WEIGHT = float(os.getenv("BM25_WEIGHT", "0.5"))
    DENSE_WEIGHT = float(os.getenv("DENSE_WEIGHT", "0.5"))
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "ibm-granite/granite-embedding-278m-multilingual")
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
    VECTOR_STORE_DIR = os.getenv("VECTOR_STORE_DIR", "vector_stores")
    MAX_CACHE_SIZE = int(os.getenv("MAX_CACHE_SIZE", "100"))
    RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "hybrid")  # Can be 'hybrid', 'dense', 'bm25', or 'all'
    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.75"))
    MIN_CHUNKS_PER_SOURCE = int(os.getenv("MIN_CHUNKS_PER_SOURCE", "3"))
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    MIN_RESUME_LENGTH = int(os.getenv("MIN_RESUME_LENGTH", "200"))  # Minimum valid resume length
    PDF_RETRY_COUNT = int(os.getenv("PDF_RETRY_COUNT", "3"))  # Number of retries for PDF extraction
    
config = Config()

# Set log level from configuration
logger.setLevel(getattr(logging, config.LOG_LEVEL))

# ------------------ Cache Management ------------------
class LRUCache:
    """Simple LRU cache implementation with capacity limit."""
    
    def __init__(self, capacity: int):
        """Initialize the cache with a given capacity."""
        self.capacity = capacity
        self.cache = {}
        self.order = []
    
    def get(self, key: str) -> Any:
        """Get an item from the cache, or None if it doesn't exist."""
        if key not in self.cache:
            return None
        
        # Move to the end (most recently used)
        self.order.remove(key)
        self.order.append(key)
        
        return self.cache[key]
    
    def put(self, key: str, value: Any) -> None:
        """Add an item to the cache, evicting the least recently used if necessary."""
        if key in self.cache:
            # Item already exists, update it and move to the end
            self.cache[key] = value
            self.order.remove(key)
            self.order.append(key)
            return
        
        # Check if cache is full
        if len(self.cache) >= self.capacity:
            # Evict the least recently used item
            lru_key = self.order[0]
            del self.cache[lru_key]
            self.order.pop(0)
        
        # Add the new item
        self.cache[key] = value
        self.order.append(key)
    
    def clear(self) -> None:
        """Clear the cache."""
        self.cache.clear()
        self.order.clear()
    
    def __len__(self) -> int:
        """Return the number of items in the cache."""
        return len(self.cache)

# Initialize global caches
vector_store_cache = LRUCache(config.MAX_CACHE_SIZE)
retriever_cache = LRUCache(config.MAX_CACHE_SIZE)
document_hash_cache = LRUCache(config.MAX_CACHE_SIZE * 2)  # Store document hashes

# ------------------ Flask App Setup ------------------
app = Flask(__name__)
CORS(app)

# ------------------ PDF Generator Dataclass ------------------
@dataclass
class PDFGenerator:
    """Represents a PDF generation method with priority."""
    name: str
    available: bool = False
    priority: int = 0

# ------------------ PDF Generation Setup ------------------
pdf_generators: List[PDFGenerator] = []

def add_pdf_generator(generator: PDFGenerator) -> None:
    """Add an available PDF generator to the list."""
    if generator.available:
        pdf_generators.append(generator)
        logger.info(f"{generator.name} is available for PDF generation")

# Check for pdflatex availability
pdflatex_gen = PDFGenerator(name="pdflatex", priority=1)
try:
    result = subprocess.run(
        ['pdflatex', '--version'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=3
    )
    pdflatex_gen.available = (result.returncode == 0)
except (subprocess.SubprocessError, FileNotFoundError):
    logger.warning("pdflatex is not available")
add_pdf_generator(pdflatex_gen)

# Check for ReportLab
try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    reportlab_gen = PDFGenerator(name="reportlab", priority=2, available=True)
    add_pdf_generator(reportlab_gen)
except ImportError:
    logger.warning("ReportLab is not available")

# Always add error generator as fallback
error_gen = PDFGenerator(name="error_pdf", priority=99, available=True)
add_pdf_generator(error_gen)
pdf_generators.sort(key=lambda x: x.priority)

# ------------------ Utility Functions ------------------
def compute_hash(text: str) -> str:
    """Compute a stable hash for a text string."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def compute_collection_id(resume_text: str, job_description: str) -> str:
    """Compute a unique collection ID based on content hashes."""
    combined_hash = compute_hash(resume_text + job_description)
    return f"collection_{combined_hash[:12]}"

def get_document_hash(text: str, source: str) -> str:
    """Get a hash for a document, with caching."""
    cache_key = f"{source}:{compute_hash(text[:100])}"  # Use first 100 chars as a quick check
    cached_hash = document_hash_cache.get(cache_key)
    
    if cached_hash:
        return cached_hash
    
    full_hash = compute_hash(text)
    document_hash_cache.put(cache_key, full_hash)
    return full_hash

def detect_language(text: str) -> str:
    """
    Attempt to detect the primary language of the text.
    Returns a language code or 'en' as default.
    """
    try:
        # Look for language identifiers
        if re.search(r'\b(python|javascript|java|c\+\+|ruby|php|golang|rust|typescript)\b', text.lower()):
            return 'code'
        # Default to English for now - can be expanded with NLP libraries if needed
        return 'en'
    except Exception as e:
        logger.error(f"Language detection error: {str(e)}")
        return 'en'

def calculate_text_statistics(text: str) -> Dict[str, Any]:
    """
    Calculate various statistics about the text to help validate extraction.
    """
    if not text:
        return {
            'length': 0,
            'word_count': 0,
            'line_count': 0,
            'avg_line_length': 0,
            'char_distribution': {},
        }
    
    words = re.findall(r'\b\w+\b', text)
    lines = text.split('\n')
    non_empty_lines = [line for line in lines if line.strip()]
    
    # Character distribution (simplified)
    char_counts = {}
    for char in text:
        if char in char_counts:
            char_counts[char] += 1
        else:
            char_counts[char] = 1
    
    # Sort by frequency
    char_distribution = dict(sorted(
        char_counts.items(), 
        key=lambda item: item[1], 
        reverse=True
    )[:10])  # Just keep top 10
    
    return {
        'length': len(text),
        'word_count': len(words),
        'line_count': len(non_empty_lines),
        'avg_line_length': sum(len(line) for line in non_empty_lines) / max(len(non_empty_lines), 1),
        'char_distribution': char_distribution,
    }

def validate_extracted_text(text: str, source_type: str = "document") -> Tuple[bool, str]:
    """
    Validate that extracted text is of reasonable quality.
    Returns (is_valid, reason).
    """
    if not text:
        return False, "No text extracted"
    
    stats = calculate_text_statistics(text)
    
    # Minimum length check
    if stats['length'] < config.MIN_RESUME_LENGTH and source_type == "resume":
        return False, f"Extracted text too short: {stats['length']} chars (min: {config.MIN_RESUME_LENGTH})"
    
    # Check for garbage text (high concentration of unusual characters)
    unusual_char_ratio = sum(stats['char_distribution'].get(c, 0) for c in '§±®©¥£€¢~`|<>[]{}') / max(stats['length'], 1)
    if unusual_char_ratio > 0.05:  # More than 5% unusual chars
        return False, f"High concentration of unusual characters: {unusual_char_ratio:.2%}"
    
    # Check for repetitive content which might indicate extraction issues
    if stats['word_count'] > 0:
        unique_words = len(set(re.findall(r'\b\w+\b', text.lower())))
        if unique_words / stats['word_count'] < 0.3 and stats['word_count'] > 20:
            return False, f"Low vocabulary diversity: {unique_words}/{stats['word_count']} unique words"
    
    return True, "Valid text"

# ------------------ Embeddings Initialization ------------------
def initialize_embeddings():
    """
    Initialize the embedding model with proper CUDA detection.
    Returns None if embeddings initialization fails completely.
    Also performs thorough tests to verify that embeddings work correctly.
    """
    logger.info(f"Initializing embeddings model: {config.EMBEDDING_MODEL}")
    
    try:
        # Check if CUDA is actually available, regardless of the config setting
        import torch
        cuda_available = torch.cuda.is_available()
        
        if config.USE_GPU and cuda_available:
            device = 'cuda'
            logger.info("CUDA is available - using GPU for embeddings")
        else:
            device = 'cpu'
            if config.USE_GPU:
                logger.warning("CUDA was requested but is not available - falling back to CPU for embeddings")
            else:
                logger.info("Using CPU for embeddings as configured")
        
        # Define model kwargs with the correct device
        model_kwargs = {
            'device': device,
        }
        
        # Create the embedding model with proper configuration
        embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs=model_kwargs,
            encode_kwargs={'normalize_embeddings': True}  # Important for cosine similarity
        )
        
        # Perform a thorough test of the embeddings
        try:
            # Test with multiple different phrases to ensure we're getting different embeddings
            test_texts = ["Resume skills and experience", "Job requirements", "Machine learning", "Software engineering"]
            test_embeddings = []
            
            for test_text in test_texts:
                embedding = embeddings.embed_query(test_text)
                test_embeddings.append(embedding)
                logger.info(f"Embedded test text: '{test_text}', vector size: {len(embedding)}")
            
            # Verify that we're getting different embeddings for different texts
            # Simple check: compare first 5 values of each embedding
            different_embeddings = False
            for i in range(1, len(test_embeddings)):
                if test_embeddings[0][:5] != test_embeddings[i][:5]:
                    different_embeddings = True
                    break
            
            if not different_embeddings:
                logger.warning("Embeddings test failed: all test queries produced similar embeddings")
                return None
            
            logger.info(f"Embeddings test successful: different texts produce different embeddings")
            return embeddings
        except Exception as e:
            logger.error(f"Embeddings function test failed: {str(e)}")
            return None
    except Exception as e:
        logger.error(f"Error initializing embeddings: {str(e)}", exc_info=True)
        logger.warning("Will proceed without embeddings and use full text in prompts instead")
        return None

# Initialize embeddings - this may return None if it fails
embeddings = initialize_embeddings()

# ------------------ Document Processing ------------------
def extract_text_from_pdf(pdf_content, document_name="resume.pdf", max_retries=None):
    """
    Extract text from PDF with enhanced reliability and validation.
    Implements retries and multiple extraction methods.
    Returns extracted text or None if extraction fails.
    """
    if max_retries is None:
        max_retries = config.PDF_RETRY_COUNT
    
    # Create a unique identifier for logging
    process_id = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
    logger.info(f"[PDF-{process_id}] Starting PDF text extraction for {document_name}")
    
    extraction_methods = [
        ("PyPDF2 standard", extract_with_pypdf2),
        ("PyPDF2 with layout", extract_with_pypdf2_layout),
    ]
    
    all_extracted_texts = []
    
    try:
        # Handle data URL format from frontend
        if isinstance(pdf_content, str) and pdf_content.startswith('data:application/pdf;base64,'):
            # Strip the prefix and decode
            logger.info(f"[PDF-{process_id}] Detected base64 data URL format, length: {len(pdf_content)}")
            base64_data = pdf_content.replace('data:application/pdf;base64,', '')
            pdf_bytes = base64.b64decode(base64_data)
            logger.info(f"[PDF-{process_id}] Decoded base64 PDF data, size: {len(pdf_bytes)} bytes")
        # Process PDF content based on type
        elif isinstance(pdf_content, bytes):
            pdf_bytes = pdf_content
        elif isinstance(pdf_content, dict) and 'bytes' in pdf_content:
            # Handle dictionary format from document
            pdf_bytes = pdf_content['bytes']
        elif hasattr(pdf_content, 'read'):
            # Handle file-like objects
            pdf_bytes = pdf_content.read()
            pdf_content.seek(0)  # Reset file pointer
        else:
            logger.error(f"[PDF-{process_id}] Unsupported PDF content type: {type(pdf_content)}")
            return None
        
        # Save PDF to a temporary file for multiple extraction attempts
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
            temp_path = temp_file.name
            temp_file.write(pdf_bytes)
        
        logger.info(f"[PDF-{process_id}] PDF saved temporarily at: {temp_path}")
        
        # Try extraction with different methods
        for method_name, extraction_func in extraction_methods:
            for attempt in range(max_retries):
                try:
                    logger.info(f"[PDF-{process_id}] Trying {method_name}, attempt {attempt+1}/{max_retries}")
                    extracted_text = extraction_func(temp_path)
                    
                    if extracted_text:
                        # Validate the extracted text
                        is_valid, reason = validate_extracted_text(extracted_text, "resume")
                        if is_valid:
                            logger.info(f"[PDF-{process_id}] Successfully extracted valid text with {method_name}")
                            
                            # Log sample of the text for debugging
                            sample = extracted_text[:500].replace('\n', ' ')
                            logger.info(f"[PDF-{process_id}] Extracted text sample: {sample}...")
                            
                            # Store for comparison
                            all_extracted_texts.append((extracted_text, method_name))
                        else:
                            logger.warning(f"[PDF-{process_id}] {method_name} produced invalid text: {reason}")
                except Exception as e:
                    logger.warning(f"[PDF-{process_id}] {method_name} extraction failed on attempt {attempt+1}: {str(e)}")
        
        # Clean up temporary file
        try:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        except:
            pass
        
        # Pick the best extraction result
        if all_extracted_texts:
            # Sort by length (descending) - usually longer is better
            all_extracted_texts.sort(key=lambda x: len(x[0]), reverse=True)
            best_text, method = all_extracted_texts[0]
            
            # Calculate stats for logging
            stats = calculate_text_statistics(best_text)
            logger.info(f"[PDF-{process_id}] Using {method} result: {stats['length']} chars, {stats['word_count']} words")
            
            return best_text
        else:
            logger.error(f"[PDF-{process_id}] All extraction methods failed")
            return None
    except Exception as e:
        logger.exception(f"[PDF-{process_id}] Error extracting text from PDF: {str(e)}")
        return None

def extract_with_pypdf2(pdf_path):
    """Extract text using standard PyPDF2 method."""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_path)
        num_pages = len(pdf_reader.pages)
        
        full_text = ""
        for i, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            if page_text:
                full_text += page_text + "\n\n"
        
        return full_text.strip()
    except Exception as e:
        logger.error(f"PyPDF2 standard extraction error: {str(e)}")
        return None

def extract_with_pypdf2_layout(pdf_path):
    """Extract text using PyPDF2 with attempt to preserve layout."""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_path)
        num_pages = len(pdf_reader.pages)
        
        full_text = ""
        for i, page in enumerate(pdf_reader.pages):
            # Get raw text
            raw_text = page.extract_text()
            if not raw_text:
                continue
                
            # Simple heuristic to preserve some layout:
            # Split by newlines, and try to detect if we need to preserve or join lines
            lines = raw_text.split('\n')
            processed_lines = []
            
            for j, line in enumerate(lines):
                if j > 0 and line.strip() and lines[j-1].strip():
                    # Check if previous line ends with indicators that the next line is a continuation
                    prev = lines[j-1].strip()
                    if (prev.endswith(',') or prev.endswith('-') or 
                        (not prev.endswith('.') and not prev.endswith(':') and len(prev) > 50)):
                        # Likely continuation - append to previous line
                        processed_lines[-1] += ' ' + line
                    else:
                        # Likely new line - keep separate
                        processed_lines.append(line)
                else:
                    processed_lines.append(line)
            
            page_text = '\n'.join(processed_lines)
            full_text += page_text + "\n\n"
        
        return full_text.strip()
    except Exception as e:
        logger.error(f"PyPDF2 layout extraction error: {str(e)}")
        return None

def create_chunks(text: str, source: str) -> List[Document]:
    """
    Split text into chunks with metadata.
    Enhanced to ensure chunks contain meaningful content.
    Adds unique document ID for caching and tracking.
    """
    if not text or len(text.strip()) < 100:
        logger.warning(f"Text for {source} is too short ({len(text) if text else 0} chars)")
        # Create at least one document to avoid breaking the pipeline
        doc_id = get_document_hash(text or "", source)
        return [Document(page_content=text or "", metadata={"source": source, "doc_id": doc_id})]
    
    logger.info(f"Creating chunks from {source} text ({len(text)} chars)")
    
    # Generate document ID for caching purposes
    doc_id = get_document_hash(text, source)
    
    # Detect language to aid in chunking
    language = detect_language(text)
    
    # Customize separators based on document type and language
    if source == "resume":
        # For resumes, prioritize section breaks and bullet points
        separators = ["\n\n", "\n", "• ", "- ", ". ", ", ", " ", ""]
    elif language == "code":
        # For code-heavy documents
        separators = ["\n\n", "\n", "; ", "} ", "{ ", ". ", ", ", " ", ""]
    else:
        # Default separators
        separators = ["\n\n", "\n", ". ", ", ", " ", ""]
    
    # Use separators that better preserve context
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=separators,
        keep_separator=True
    )
    
    # Split text into chunks
    chunks = text_splitter.split_text(text)
    
    # Create document objects with metadata
    documents = []
    for i, chunk in enumerate(chunks):
        # Skip empty or extremely short chunks
        if len(chunk.strip()) < 20:
            logger.warning(f"Skipping short chunk ({len(chunk)} chars)")
            continue
            
        chunk_id = f"{doc_id}_{i}"
        documents.append(Document(
            page_content=chunk,
            metadata={
                "source": source,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "language": language
            }
        ))
    
    # Log some information about the chunks
    if documents:
        logger.info(f"Created {len(documents)} chunks for {source}")
        logger.info(f"Sample chunk: '{documents[0].page_content[:100]}...'")
    else:
        logger.warning(f"No valid chunks created for {source}")
        
    return documents

# ------------------ Vector Store Management ------------------
def get_collection_path(collection_id: str) -> str:
    """Get path for a specific collection's vector store using content-based ID."""
    # Ensure the directory exists
    path = os.path.join(config.VECTOR_STORE_DIR, collection_id)
    os.makedirs(path, exist_ok=True)
    
    return path

def create_or_load_vector_store(documents: List[Document], collection_id: str) -> Optional[Chroma]:
    """
    Create a vector store from documents or load from cache if exists.
    Enhanced with better caching and content-based IDs.
    """
    if not embeddings:
        logger.warning("No embeddings available - skipping vector store creation")
        return None
        
    if not documents:
        logger.error("No documents provided for vector store creation")
        return None
    
    # Check if we have this vector store cached
    cached_store = vector_store_cache.get(collection_id)
    if cached_store:
        logger.info(f"Using cached vector store for collection {collection_id}")
        return cached_store
    
    # Check if the store exists on disk
    persist_directory = get_collection_path(collection_id)
    collection_exists = os.path.exists(persist_directory) and len(os.listdir(persist_directory)) > 0
    
    try:
        if collection_exists:
            logger.info(f"Loading existing vector store from {persist_directory}")
            vector_store = Chroma(
                collection_name=collection_id,
                embedding_function=embeddings,
                persist_directory=persist_directory
            )
            
            # Validate the loaded store
            test_results = vector_store.similarity_search("test query", k=1)
            logger.info(f"Vector store validation successful: retrieved {len(test_results)} results")
            
            # Cache for future use
            vector_store_cache.put(collection_id, vector_store)
            return vector_store
    except Exception as load_error:
        logger.warning(f"Error loading existing vector store: {str(load_error)}")
        logger.info("Will create a new vector store instead")
        # Continue to creation below
    
    # Create a new vector store
    logger.info(f"Creating new vector store with {len(documents)} documents")
    try:
        # Generate a unique path for this vector store based on content
        persist_directory = get_collection_path(collection_id)
        logger.info(f"Vector store will be persisted at: {persist_directory}")
        
        # Create the vector store with proper parameters
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=persist_directory,
            collection_name=collection_id
        )
        
        # Persist to disk
        vector_store.persist()
        
        # Only do full validation on new vector stores
        validate_vector_store(vector_store, documents)
        
        # Cache for future use
        vector_store_cache.put(collection_id, vector_store)
        
        return vector_store
    except Exception as e:
        logger.error(f"Error creating vector store: {str(e)}")
        return None

def validate_vector_store(vector_store: Chroma, documents: List[Document]) -> bool:
    """
    Validate a vector store with thorough testing.
    Only called for new vector stores to avoid redundant testing.
    """
    try:
        logger.info("Validating new vector store")
        
        # Extract some representative text from documents to use as test queries
        test_queries = []
        for doc in documents[:min(3, len(documents))]:
            # Extract potentially meaningful words (nouns, etc.)
            words = re.findall(r'\b[A-Za-z]{4,}\b', doc.page_content)
            if words:
                # Pick the middle word to avoid boilerplate at beginning/end
                middle_idx = len(words) // 2
                test_queries.append(words[middle_idx])
        
        # Add generic test query if we couldn't extract good ones
        if not test_queries:
            test_queries = ["skills", "experience", "requirements"]
            
        # Test with all queries
        all_tests_passed = True
        for query in test_queries:
            results = vector_store.similarity_search(query, k=2)
            if not results:
                logger.warning(f"Vector store test returned no results for query: '{query}'")
                all_tests_passed = False
            else:
                logger.info(f"Vector store test successful for query '{query}': returned {len(results)} results")
                logger.info(f"First result: '{results[0].page_content[:50]}...'")
        
        return all_tests_passed
    except Exception as e:
        logger.error(f"Vector store validation failed: {str(e)}")
        return False

def create_hybrid_retriever(
    resume_chunks: List[Document], 
    job_chunks: List[Document], 
    vector_store: Optional[Chroma],
    collection_id: str
) -> Optional[Any]:
    """
    Create a retriever based on the configuration.
    Can create BM25-only, dense-only, or hybrid retrievers.
    Implements caching and per-source retrieval options.
    """
    all_chunks = resume_chunks + job_chunks
    
    if not all_chunks:
        logger.error("No documents provided for retriever creation")
        return None
    
    # Check if we have this retriever cached
    cached_retriever = retriever_cache.get(collection_id)
    if cached_retriever:
        logger.info(f"Using cached retriever for collection {collection_id}")
        return cached_retriever
    
    retrieval_mode = config.RETRIEVAL_MODE.lower()
    logger.info(f"Creating retriever in {retrieval_mode} mode")
    
    try:
        # Create BM25 retriever for keyword search
        bm25_retriever = None
        if retrieval_mode in ['bm25', 'hybrid', 'all']:
            bm25_retriever = BM25Retriever.from_documents(all_chunks)
            bm25_retriever.k = 5  # Number of documents to retrieve
            logger.info("Created BM25 retriever")
            
            # Test BM25 retriever
            try:
                test_query = "skills requirements experience"
                bm25_results = bm25_retriever.get_relevant_documents(test_query)
                logger.info(f"BM25 retriever test: {len(bm25_results)} results for query '{test_query}'")
                
                # If in BM25-only mode, cache and return it
                if retrieval_mode == 'bm25':
                    retriever_cache.put(collection_id, bm25_retriever)
                    return bm25_retriever
            except Exception as bm25_error:
                logger.error(f"BM25 retriever test failed: {str(bm25_error)}")
                if retrieval_mode == 'bm25':
                    return None
        
        # If we need to use dense retrieval but have no vector store, return BM25 or None
        if not vector_store and retrieval_mode in ['dense', 'hybrid', 'all']:
            logger.warning("No vector store available for dense retrieval")
            return bm25_retriever
        
        # Create vector retriever for semantic search
        vector_retriever = None
        if retrieval_mode in ['dense', 'hybrid', 'all'] and vector_store:
            vector_retriever = vector_store.as_retriever(search_kwargs={"k": 5})
            logger.info("Created vector retriever")
            
            # Test vector retriever
            try:
                test_query = "skills requirements experience"
                vector_results = vector_retriever.get_relevant_documents(test_query)
                logger.info(f"Vector retriever test: {len(vector_results)} results for query '{test_query}'")
                
                # If in dense-only mode, cache and return it
                if retrieval_mode == 'dense':
                    retriever_cache.put(collection_id, vector_retriever)
                    return vector_retriever
            except Exception as vector_error:
                logger.error(f"Vector retriever test failed: {str(vector_error)}")
                if retrieval_mode == 'dense':
                    return None
        
        # Create the appropriate retriever based on the mode
        final_retriever = None
        
        if retrieval_mode == 'hybrid' and bm25_retriever and vector_retriever:
            # Create hybrid retriever with normalized weights
            hybrid_retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, vector_retriever],
                weights=[config.BM25_WEIGHT, config.DENSE_WEIGHT]
            )
            logger.info(f"Created hybrid retriever with weights: BM25={config.BM25_WEIGHT}, Dense={config.DENSE_WEIGHT}")
            final_retriever = hybrid_retriever
        elif retrieval_mode == 'all' and bm25_retriever and vector_retriever:
            # Special mode: combine results from both methods but separately
            # We'll handle this in the retrieval function
            logger.info("Using 'all' retrieval mode - will query both retrievers separately")
            final_retriever = {
                'bm25': bm25_retriever,
                'vector': vector_retriever
            }
        elif bm25_retriever:
            final_retriever = bm25_retriever
        elif vector_retriever:
            final_retriever = vector_retriever
        else:
            logger.error("Failed to create any retriever")
            return None
        
        # Cache and return the retriever
        retriever_cache.put(collection_id, final_retriever)
        return final_retriever
    except Exception as e:
        logger.error(f"Error creating retriever: {str(e)}")
        return None

def filter_and_balance_chunks(
    retrieved_chunks: List[Document], 
    resume_chunks: List[Document], 
    job_chunks: List[Document]
) -> List[Document]:
    """
    Filter chunks for relevance and balance between resume and job description.
    Ensures minimum representation from both sources.
    Performs deduplication and relevance filtering.
    """
    if not retrieved_chunks:
        logger.warning("No chunks were retrieved, will use default selection")
        # Select some default chunks if retrieval failed
        result = []
        if resume_chunks:
            result.extend(resume_chunks[:min(config.MIN_CHUNKS_PER_SOURCE, len(resume_chunks))])
        if job_chunks:
            result.extend(job_chunks[:min(config.MIN_CHUNKS_PER_SOURCE, len(job_chunks))])
        return result
    
    # First, count chunks from each source
    resume_retrieved = [chunk for chunk in retrieved_chunks if chunk.metadata.get("source") == "resume"]
    job_retrieved = [chunk for chunk in retrieved_chunks if chunk.metadata.get("source") == "job"]
    
    logger.info(f"Initial retrieved chunks: {len(resume_retrieved)} from resume, {len(job_retrieved)} from job")
    
    result = list(retrieved_chunks)  # Start with all retrieved chunks
    
    # Ensure minimum number of chunks from each source
    if len(resume_retrieved) < config.MIN_CHUNKS_PER_SOURCE and resume_chunks:
        # Find chunks that aren't already included
        existing_ids = {chunk.metadata.get("chunk_id") for chunk in resume_retrieved if "chunk_id" in chunk.metadata}
        additional_chunks = []
        
        for chunk in resume_chunks:
            if len(additional_chunks) >= (config.MIN_CHUNKS_PER_SOURCE - len(resume_retrieved)):
                break
            if chunk.metadata.get("chunk_id") not in existing_ids:
                additional_chunks.append(chunk)
                existing_ids.add(chunk.metadata.get("chunk_id"))
        
        if additional_chunks:
            logger.info(f"Adding {len(additional_chunks)} additional resume chunks to meet minimum")
            result.extend(additional_chunks)
    
    if len(job_retrieved) < config.MIN_CHUNKS_PER_SOURCE and job_chunks:
        # Find chunks that aren't already included
        existing_ids = {chunk.metadata.get("chunk_id") for chunk in job_retrieved if "chunk_id" in chunk.metadata}
        additional_chunks = []
        
        for chunk in job_chunks:
            if len(additional_chunks) >= (config.MIN_CHUNKS_PER_SOURCE - len(job_retrieved)):
                break
            if chunk.metadata.get("chunk_id") not in existing_ids:
                additional_chunks.append(chunk)
                existing_ids.add(chunk.metadata.get("chunk_id"))
        
        if additional_chunks:
            logger.info(f"Adding {len(additional_chunks)} additional job chunks to meet minimum")
            result.extend(additional_chunks)
    
    # Deduplicate by chunk_id
    seen_ids = set()
    deduplicated = []
    
    for chunk in result:
        chunk_id = chunk.metadata.get("chunk_id")
        if not chunk_id or chunk_id not in seen_ids:
            if chunk_id:
                seen_ids.add(chunk_id)
            deduplicated.append(chunk)
    
    if len(deduplicated) < len(result):
        logger.info(f"Removed {len(result) - len(deduplicated)} duplicate chunks")
    
    return deduplicated

def generate_retrieval_queries(resume_chunks: List[Document], job_chunks: List[Document]) -> List[str]:
    """
    Generate custom queries based on resume and job content.
    Creates queries that are specific to the content rather than generic.
    """
    queries = []
    
    # Start with some standard queries
    standard_queries = [
        "skills experience qualifications requirements",
        "technical expertise professional background",
        "job responsibilities duties skills"
    ]
    queries.extend(standard_queries)
    
    # Try to extract job title and key terms from the job chunks
    job_title = None
    key_terms = set()
    
    # Simple extraction of potential job title from the first few chunks
    if job_chunks:
        for chunk in job_chunks[:2]:
            # Look for common job title patterns
            title_patterns = [
                r'(?:Job Title|Position):\s*([A-Za-z0-9 ]+(?:Developer|Engineer|Manager|Analyst|Designer|Architect|Consultant|Specialist|Director|Lead))',
                r'(?:Hiring|Seeking|Looking for)[^.]*?([A-Za-z0-9 ]+(?:Developer|Engineer|Manager|Analyst|Designer|Architect|Consultant|Specialist|Director|Lead))',
                r'([A-Za-z0-9 ]+(?:Developer|Engineer|Manager|Analyst|Designer|Architect|Consultant|Specialist|Director|Lead))\s*(?:position|role|job)'
            ]
            
            for pattern in title_patterns:
                match = re.search(pattern, chunk.page_content)
                if match:
                    job_title = match.group(1).strip()
                    break
            
            if job_title:
                break
    
    # Extract key terms from job description
    if job_chunks:
        # Look for skills, technologies, tools
        for chunk in job_chunks[:4]:  # Limit to first few chunks
            # Extract potential skill terms
            skill_matches = re.finditer(r'\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)*|[A-Z]{2,}(?:\+\+)?|[A-Za-z0-9]+(?:\.[A-Za-z0-9]+)+)\b', chunk.page_content)
            for match in skill_matches:
                term = match.group(1)
                if len(term) > 2 and not term.lower() in ['the', 'and', 'for', 'with', 'this']:
                    key_terms.add(term)
    
    # Create custom queries using job title and key terms
    if job_title:
        queries.append(f"{job_title} skills requirements experience")
        queries.append(f"qualifications for {job_title} position")
    
    # Add key terms to queries
    if key_terms:
        # Convert set to list for better handling
        term_list = list(key_terms)
        
        # Take up to 10 terms to avoid overly long queries
        for i in range(0, min(len(term_list), 10), 3):
            terms_subset = term_list[i:i+3]
            queries.append(f"{' '.join(terms_subset)} experience skills")
    
    # Make queries unique
    unique_queries = list(dict.fromkeys(queries))
    
    return unique_queries

def retrieve_relevant_chunks(
    retriever: Any, 
    resume_chunks: List[Document], 
    job_chunks: List[Document]
) -> List[Document]:
    """
    Perform retrieval with dynamic query generation and source balancing.
    Supports different retrieval modes and handles special 'all' mode.
    """
    if not retriever:
        logger.warning("No retriever available, using default chunks")
        return filter_and_balance_chunks([], resume_chunks, job_chunks)
    
    # Generate dynamic queries based on the content
    queries = generate_retrieval_queries(resume_chunks, job_chunks)
    logger.info(f"Using {len(queries)} dynamic queries for retrieval")
    
    all_retrieved_chunks = []
    
    try:
        # Handle different retriever types
        if isinstance(retriever, dict) and 'bm25' in retriever and 'vector' in retriever:
            # Special 'all' mode - query both retrievers separately
            logger.info("Querying both BM25 and vector retrievers separately")
            
            for query in queries:
                # Query BM25 retriever
                try:
                    bm25_results = retriever['bm25'].get_relevant_documents(query)
                    logger.info(f"BM25 retrieved {len(bm25_results)} chunks for query: '{query[:30]}...'")
                    all_retrieved_chunks.extend(bm25_results)
                except Exception as bm25_error:
                    logger.error(f"BM25 retrieval failed for query '{query[:30]}...': {str(bm25_error)}")
                
                # Query vector retriever
                try:
                    vector_results = retriever['vector'].get_relevant_documents(query)
                    logger.info(f"Vector retrieved {len(vector_results)} chunks for query: '{query[:30]}...'")
                    all_retrieved_chunks.extend(vector_results)
                except Exception as vector_error:
                    logger.error(f"Vector retrieval failed for query '{query[:30]}...': {str(vector_error)}")
        else:
            # Standard retriever (BM25, dense, or hybrid)
            for query in queries:
                try:
                    results = retriever.get_relevant_documents(query)
                    logger.info(f"Retrieved {len(results)} chunks for query: '{query[:30]}...'")
                    all_retrieved_chunks.extend(results)
                except Exception as e:
                    logger.error(f"Retrieval failed for query '{query[:30]}...': {str(e)}")
    except Exception as e:
        logger.error(f"Error during retrieval: {str(e)}")
    
    # Filter and balance the chunks
    filtered_chunks = filter_and_balance_chunks(all_retrieved_chunks, resume_chunks, job_chunks)
    
    logger.info(f"After filtering and balancing: {len(filtered_chunks)} chunks")
    return filtered_chunks

# ------------------ LLM Analysis Functions ------------------
def analyze_with_llm(resume_text: str, job_description: str, model: str) -> Dict[str, Any]:
    """
    Use the LLM to analyze the resume against the job description.
    Enhanced with better prompting for structured output and error handling.
    """
    # Create a prompt that works well with structured output
    prompt = f"""
# Resume and Job Analysis Task

Analyze this resume and job description to extract key information and determine how well they match.

## Resume:
```
{resume_text[:3000]}  
```
{f"[Resume continues, total length: {len(resume_text)} characters]" if len(resume_text) > 3000 else ""}

## Job Description:
```
{job_description[:3000]}
```
{f"[Job description continues, total length: {len(job_description)} characters]" if len(job_description) > 3000 else ""}

## Analysis Instructions:
1. Identify skills mentioned in both the resume and job description
2. List skills in the job description that are not in the resume
3. Extract education details from the resume
4. Extract experience details from the resume
5. Identify key responsibilities from the job description
6. Calculate an approximate match percentage (0-100%)

Format your response as a valid JSON object with these exact keys:
- matching_skills (array of strings)
- missing_skills (array of strings)
- education (array of objects with degree, institution, year)
- experience (array of objects with title, company, dates)
- responsibilities (array of strings)
- match_percentage (number)

Here's the exact JSON format to follow:
```json
{{
  "matching_skills": ["skill1", "skill2"],
  "missing_skills": ["skill3", "skill4"],
  "education": [
    {{ "degree": "Bachelor's", "institution": "University Name", "year": "2019" }}
  ],
  "experience": [
    {{ "title": "Job Title", "company": "Company Name", "dates": "Jan 2020 - Present" }}
  ],
  "responsibilities": ["responsibility1", "responsibility2"],
  "match_percentage": 75
}}
```

Return ONLY the JSON object, with no additional text before or after.
"""

    try:
        # Call Ollama API for analysis
        logger.info(f"Calling Ollama for resume analysis with model: {model}")
        start_time = time.time()
        
        response = ollama.generate(
            model=model,
            prompt=prompt,
            options={
                "temperature": 0.01,  # Keep temperature low for structured output
                "num_predict": 2048,
            }
        )
        
        elapsed_time = time.time() - start_time
        logger.info(f"LLM response received in {elapsed_time:.2f} seconds")
        
        # Extract and parse the JSON response
        output = response.get('response', '')
        
        # Clean up the output to extract just the JSON part
        # Look for the JSON structure more robustly
        output = output.strip()
        
        # First try to extract the most likely JSON block
        json_match = re.search(r'(\{.*\})', output, re.DOTALL)
        if json_match:
            output = json_match.group(1)
        
        # If starts with "```json" and ends with "```", extract just the content
        if output.startswith("```json") and "```" in output[7:]:
            output = output[7:].split("```", 1)[0].strip()
        
        # If starts with "```" and ends with "```", extract just the content
        elif output.startswith("```") and output.endswith("```"):
            output = output[3:-3].strip()
        
        try:
            logger.info("Parsing JSON response")
            analysis = json.loads(output)
            
            # Validate the structure of the response
            required_keys = ['matching_skills', 'missing_skills', 'education', 
                             'experience', 'responsibilities', 'match_percentage']
            
            for key in required_keys:
                if key not in analysis:
                    analysis[key] = [] if key != 'match_percentage' else 0
            
            # Ensure match_percentage is a number
            if not isinstance(analysis['match_percentage'], (int, float)):
                try:
                    analysis['match_percentage'] = float(analysis['match_percentage'])
                except (ValueError, TypeError):
                    analysis['match_percentage'] = 0
                
            logger.info(f"Successfully parsed LLM analysis with match percentage: {analysis['match_percentage']}%")
            return analysis
            
        except json.JSONDecodeError as je:
            logger.error(f"Failed to parse JSON from LLM response: {str(je)}")
            logger.error(f"Raw LLM output: {output[:500]}...")
            
            # Make a better attempt to fix common JSON issues
            try:
                # Replace single quotes with double quotes
                fixed_output = output.replace("'", '"')
                # Try again with the fixed output
                analysis = json.loads(fixed_output)
                logger.info("Successfully parsed JSON after fixing quotes")
                
                # Validate required keys
                for key in required_keys:
                    if key not in analysis:
                        analysis[key] = [] if key != 'match_percentage' else 0
                
                return analysis
            except Exception:
                # Return a default structure if parsing fails
                return {
                    "matching_skills": [],
                    "missing_skills": [],
                    "education": [],
                    "experience": [],
                    "responsibilities": [],
                    "match_percentage": 0,
                    "error": "Failed to parse LLM analysis",
                    "raw_output": output[:500]  # Include partial output for debugging
                }
    except Exception as e:
        logger.exception(f"Error in LLM analysis: {str(e)}")
        return {
            "matching_skills": [],
            "missing_skills": [],
            "education": [],
            "experience": [],
            "responsibilities": [],
            "match_percentage": 0,
            "error": str(e)
        }

# ------------------ Utilities ------------------
@contextmanager
def temp_file_manager(suffix: str = '.tex'):
    """Context manager for temporary file creation and cleanup."""
    temp_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    temp_path = Path(temp_file.name)
    temp_file.close()
    try:
        yield temp_path
    finally:
        try:
            if temp_path.exists():
                temp_path.unlink()
        except Exception as e:
            logger.warning(f"Failed to remove temporary file {temp_path}: {str(e)}")

def sanitize_input(text: str) -> str:
    """Sanitize text for LaTeX output."""
    # Handle None or non-string values
    if not isinstance(text, str):
        return str(text)
    
    return text.replace('&', r'\&').replace('%', r'\%').replace('_', r'\_').replace('#', r'\#').replace('$', r'\$')

def auto_wrap_links(text: str) -> str:
    """Automatically finds URLs in text and wraps them with LaTeX \href command."""
    if not isinstance(text, str):
        return str(text)
    
    url_pattern = r'(https?://[^\s]+)'
    def replace_url(match):
        url = match.group(0)
        return rf'\href{{{url}}}{{{url}}}'
    return re.sub(url_pattern, replace_url, text)

# ------------------ Prompt Generation ------------------
def generate_tailoring_prompt(resume_text: str, job_description: str, analysis: Dict, relevant_chunks: Optional[List[Document]] = None) -> str:
    """
    Generate a detailed prompt for resume tailoring with enhanced project emphasis
    and strict formatting requirements.
    """
    # Format skills lists
    matching_skills = ", ".join(analysis.get("matching_skills", []))
    missing_skills = ", ".join(analysis.get("missing_skills", []))
    
    # Format education list
    education_list = ""
    for edu in analysis.get("education", []):
        degree = edu.get("degree", "")
        institution = edu.get("institution", "")
        year = edu.get("year", "")
        education_list += f"- {degree} from {institution} ({year})\n"
    
    # Format experience list
    experience_list = ""
    for exp in analysis.get("experience", []):
        title = exp.get("title", "")
        company = exp.get("company", "")
        dates = exp.get("dates", "")
        experience_list += f"- {title} at {company} ({dates})\n"
    
    # Format responsibilities list
    responsibilities_list = ""
    for resp in analysis.get("responsibilities", []):
        responsibilities_list += f"- {resp}\n"
    
    # Extract all sections for strict preservation
    sections_dict = {}
    section_pattern = r'\\section\*{([^}]+)}(.*?)(?=\\section\*{|\\end{document}|$)'
    for title, content in re.findall(section_pattern, resume_text, re.DOTALL):
        sections_dict[title] = content.strip()
    
    # List all sections found for direct reference
    sections_list = ", ".join([f"'{section}'" for section in sections_dict.keys()])
    
    # Extract original experience entries for explicit preservation
    experience_entries = []
    if "Experience" in sections_dict:
        exp_content = sections_dict["Experience"]
        # Extract company names for direct reference
        company_pattern = r'\\item\\textbf{([^}]+)}'
        company_names = re.findall(company_pattern, exp_content)
        # Extract date patterns as well
        date_pattern = r'\\item\\textbf{([^}]+)}[^\\]*\\hfill\s*([^\\]+)'
        date_matches = re.findall(date_pattern, exp_content)
        
        for i, company in enumerate(company_names):
            date = date_matches[i][1] if i < len(date_matches) else "N/A"
            experience_entries.append(f"- {company} ({date})")
    
    # Extract original projects for special handling
    projects_content = ""
    projects_pattern = r'\\section\*{Projects}(.*?)(?=\\section\*{|\\end{document}|$)'
    projects_match = re.search(projects_pattern, resume_text, re.DOTALL)
    if projects_match:
        projects_content = projects_match.group(1).strip()
    
    # Extract project names for explicit preservation
    project_names = []
    if projects_content:
        project_pattern = r'\\item\\textbf{([^}]+)}'
        project_names = re.findall(project_pattern, projects_content)
    
    # Build prompt with enhanced project emphasis and strict formatting requirements
    prompt = f"""# Resume Tailoring Task with EXACT FORMAT AND CONTENT PRESERVATION

You are an expert resume writer who tailors resumes to specific job descriptions while PRECISELY COPYING the original resume's LaTeX formatting and content structure.

## 1. ORIGINAL LaTeX RESUME TO PRESERVE COMPLETELY
Below is the LaTeX source for the original resume. You MUST copy this formatting structure EXACTLY:

```latex
{resume_text}

{f"[Job description continues, total length: {len(job_description)} characters]" if len(job_description) > 3000 else ""}
3. Key Findings from Analysis
Match Percentage: {analysis.get('match_percentage')}%
Matching Skills: {matching_skills if matching_skills else "None found"}
Missing Skills: {missing_skills if missing_skills else "None found"}
Education:
{education_list if education_list else "None extracted"}
Experience:
{experience_list if experience_list else "None extracted"}
Key Job Responsibilities:
{responsibilities_list if responsibilities_list else "None extracted"}
4. CRITICAL FORMATTING AND CONTENT PRESERVATION INSTRUCTIONS

Copy the EXACT LaTeX formatting of the original resume
Maintain ALL sections found in the original: {sections_list}
Keep ALL of the original experience entries - DO NOT REMOVE ANY JOBS
Do not change the order of sections or their titles
DO NOT ADD ANY NEW COMPANIES OR EXPERIENCES that weren't in the original resume
Preserve ALL original Projects with their exact names and structure
Only modify the descriptions of experiences and projects to match the job description
DO NOT DUPLICATE ANY CONTENT
DO NOT INVENT OR CREATE NEW EXPERIENCE ENTRIES

5. EXPERIENCE SECTION INSTRUCTIONS - EXTREMELY IMPORTANT

YOU MUST KEEP EXACTLY THESE EXPERIENCE ENTRIES WITH THEIR EXACT COMPANY NAMES AND DATES:
{chr(10).join(experience_entries)}
DO NOT CHANGE ANY COMPANY NAMES OR DATES - only modify the bullet points under each experience
DO NOT ADD ANY NEW COMPANIES that weren't in the original resume
DO NOT REMOVE ANY COMPANIES from the original resume
Maintain the EXACT SAME ORDER of companies as in the original resume
Only modify the bullet points under each experience to better match the job requirements
NEVER invent fictional companies or experience entries

6. PROJECTS SECTION INSTRUCTIONS - EXTREMELY IMPORTANT

PRESERVE EXACTLY THESE PROJECT NAMES in their original order:
{chr(10).join([f"- {project}" for project in project_names])}
DO NOT ADD OR REMOVE ANY PROJECTS
Keep all project names and links exactly as they appear in the original
Only modify the descriptions slightly to highlight aspects relevant to the job description
If the original had bullet points under projects, maintain that exact structure
Keep all GitHub links and website references intact

7. Output Format Requirements
Your output must follow this exact format:
<TAILORED_RESUME>
\\documentclass{{article}}
\\usepackage{{geometry}}
\\usepackage{{hyperref}}
\\geometry{{margin=1in}}
\\hypersetup{{colorlinks=true, linkcolor=blue, urlcolor=blue}}
\\begin{{document}}
% COPY THE ENTIRE ORIGINAL RESUME STRUCTURE HERE
% ONLY MODIFY DESCRIPTIONS TO MATCH JOB REQUIREMENTS
\\end{{document}}
</TAILORED_RESUME>
<CHANGES_MADE>
Explain the specific changes made to tailor this resume, including:

Skills emphasized and why
Experience descriptions modified
Project descriptions enhanced
ATS optimization strategies used
</CHANGES_MADE>

REMEMBER: Your primary task is to MAINTAIN all original content while ONLY MODIFYING descriptions to better match the job description. DO NOT INVENT NEW EXPERIENCES OR COMPANIES.
"""
    return prompt


def check_for_duplicate_experiences(latex_content: str) -> Dict[str, Any]:
    """
    Check if there are duplicate experiences in the LaTeX content.
    Returns a dictionary with has_duplicates flag and list of duplicated companies.
    """
    if not latex_content or "\\section*{Experience}" not in latex_content:
        return {"has_duplicates": False, "duplicates": []}
    
    # Extract the Experience section
    exp_pattern = r'\\section\*{Experience}(.*?)(?=\\section\*{|\\end{document}|$)'
    exp_match = re.search(exp_pattern, latex_content, re.DOTALL)
    if not exp_match:
        return {"has_duplicates": False, "duplicates": []}
    
    exp_section = exp_match.group(1)
    
    # Extract company names
    company_pattern = r'\\item\\textbf{([^}]+)}'
    companies = re.findall(company_pattern, exp_section)
    
    # Check for duplicates
    seen_companies = set()
    duplicates = []
    
    for company in companies:
        if company in seen_companies:
            duplicates.append(company)
        else:
            seen_companies.add(company)
    
    return {
        "has_duplicates": len(duplicates) > 0,
        "duplicates": duplicates
    }


def fix_duplicate_experiences(tailored_latex: str, original_latex: str) -> str:
    """
    Fix duplicate experiences in tailored LaTeX content by preserving the original structure.
    """
    if not tailored_latex or "\\section*{Experience}" not in tailored_latex:
        return tailored_latex
        
    # Extract sections from original resume
    original_sections = {}
    section_pattern = r'\\section\*{([^}]+)}(.*?)(?=\\section\*{|\\end{document}|$)'
    for title, content in re.findall(section_pattern, original_latex, re.DOTALL):
        original_sections[title] = content.strip()
    
    # If there's no Experience section in the original, we can't fix it
    if "Experience" not in original_sections:
        return tailored_latex
    
    # Extract company entries from original Experience section
    original_exp = original_sections["Experience"]
    company_pattern = r'\\item\\textbf{([^}]+)}'
    original_companies = []
    seen_companies = set()
    
    for company in re.findall(company_pattern, original_exp):
        if company not in seen_companies:
            original_companies.append(company)
            seen_companies.add(company)
    
    # Extract the Experience section from tailored resume
    exp_pattern = r'(\\section\*{Experience})(.*?)(?=\\section\*{|\\end{document}|$)'
    exp_match = re.search(exp_pattern, tailored_latex, re.DOTALL)
    if not exp_match:
        return tailored_latex
    
    # Build a corrected Experience section
    corrected_exp = "\\begin{itemize}\n"
    
    # For each original company, find its content in the tailored resume (first instance only)
    tailored_exp = exp_match.group(2)
    
    for company in original_companies:
        # Find this company's entry in the tailored resume
        company_escaped = re.escape(company)
        entry_pattern = f"\\\\item\\\\textbf{{{company_escaped}}}.*?(?=\\\\item\\\\textbf{{|\\\\end{{itemize}})"
        entry_match = re.search(entry_pattern, tailored_exp, re.DOTALL)
        
        if entry_match:
            # Use the tailored content for this company
            company_entry = entry_match.group(0)
            corrected_exp += company_entry + "\n"
        else:
            # If we can't find it in the tailored resume, use the original content
            original_entry_match = re.search(
                f"\\\\item\\\\textbf{{{company_escaped}}}.*?(?=\\\\item\\\\textbf{{|\\\\end{{itemize}})",
                original_exp,
                re.DOTALL
            )
            if original_entry_match:
                company_entry = original_entry_match.group(0)
                corrected_exp += company_entry + "\n"
            else:
                # Fallback if we can't find it in either
                corrected_exp += f"\\item\\textbf{{{company}}} \\hfill DATE\\newline\nPosition\n"
    
    corrected_exp += "\\end{itemize}"
    
    # Replace the Experience section in the tailored resume
    fixed_latex = re.sub(
        exp_pattern,
        lambda m: f"{m.group(1)}\n{corrected_exp}",
        tailored_latex,
        flags=re.DOTALL
    )
    
    return fixed_latex


def check_for_experience_integrity(tailored_latex: str, original_latex: str) -> Dict[str, Any]:
    """
    Comprehensively check if the experience section in the tailored resume matches the original.
    Returns a dictionary with detailed analysis of issues.
    """
    if not tailored_latex or not original_latex:
        return {"is_valid": False, "issues": ["Missing content"]}
    
    # Extract the Experience sections
    orig_exp_pattern = r'\\section\*{Experience}(.*?)(?=\\section\*{|\\end{document}|$)'
    tailored_exp_pattern = r'\\section\*{Experience}(.*?)(?=\\section\*{|\\end{document}|$)'
    
    orig_exp_match = re.search(orig_exp_pattern, original_latex, re.DOTALL)
    tailored_exp_match = re.search(tailored_exp_pattern, tailored_latex, re.DOTALL)
    
    if not orig_exp_match or not tailored_exp_match:
        return {"is_valid": False, "issues": ["Experience section not found"]}
    
    orig_exp = orig_exp_match.group(1)
    tailored_exp = tailored_exp_match.group(1)
    
    # Extract company names
    company_pattern = r'\\item\\textbf{([^}]+)}'
    original_companies = re.findall(company_pattern, orig_exp)
    tailored_companies = re.findall(company_pattern, tailored_exp)
    
    # Issues to track
    issues = []
    company_issues = []
    
    # Check for missing companies
    orig_companies_set = set(original_companies)
    tailored_companies_set = set(tailored_companies)
    
    missing_companies = orig_companies_set - tailored_companies_set
    if missing_companies:
        issues.append(f"Missing companies: {', '.join(missing_companies)}")
        company_issues.extend(list(missing_companies))
    
    # Check for added companies (not in original)
    added_companies = tailored_companies_set - orig_companies_set
    if added_companies:
        issues.append(f"Added companies not in original: {', '.join(added_companies)}")
        for company in added_companies:
            company_issues.append(f"Remove: {company}")
    
    # Check company order
    for i, company in enumerate(original_companies):
        if i < len(tailored_companies) and company != tailored_companies[i]:
            issues.append(f"Company order changed: '{company}' should be at position {i+1}")
    
    # Check for duplicate companies
    company_counts = {}
    for company in tailored_companies:
        company_counts[company] = company_counts.get(company, 0) + 1
    
    duplicated_companies = {company: count for company, count in company_counts.items() if count > 1}
    if duplicated_companies:
        for company, count in duplicated_companies.items():
            issues.append(f"Duplicate entry: '{company}' appears {count} times")
            company_issues.append(f"Deduplicate: {company}")
    
    # Check date formats for existing companies
    date_pattern = r'\\item\\textbf{([^}]+)}[^\\]*\\hfill\s*([^\\]+)'
    orig_dates = {match[0]: match[1] for match in re.findall(date_pattern, orig_exp)}
    tailored_dates = {match[0]: match[1] for match in re.findall(date_pattern, tailored_exp)}
    
    for company, date in orig_dates.items():
        if company in tailored_dates and date != tailored_dates[company]:
            issues.append(f"Date changed for '{company}': should be '{date}' but found '{tailored_dates[company]}'")
            company_issues.append(f"Fix date: {company}")
    
    # Check for structural issues in the Experience section
    if "\\begin{itemize}" not in tailored_exp or "\\end{itemize}" not in tailored_exp:
        issues.append("Experience section missing proper itemize environment")
    
    return {
        "is_valid": len(issues) == 0,
        "issues": issues,
        "company_issues": company_issues,
        "original_companies": original_companies,
        "tailored_companies": tailored_companies
    }
    
def create_section_correction_prompt(original_latex: str, tailored_latex: str, evaluation: Dict) -> str:
    """
    Creates a targeted prompt to fix specific issues in the tailored resume.
    """
    issues = evaluation.get("issues", [])
    fix_suggestions = evaluation.get("fix_suggestions", {})
    
    # Extract sections from original for comparison
    original_sections = {}
    section_pattern = r'\\section\*{([^}]+)}(.*?)(?=\\section\*{|\\end{document}|$)'
    for title, content in re.findall(section_pattern, original_latex, re.DOTALL):
        original_sections[title] = content.strip()
    
    # Create specific instructions based on detected issues
    specific_instructions = []
    
    # Handle missing sections
    if "missing_sections" in fix_suggestions:
        missing_sections = set(original_sections.keys()) - set(re.findall(r'\\section\*{([^}]+)}', tailored_latex))
        specific_instructions.append("MISSING SECTIONS ISSUE: You must add back these missing sections exactly as they appear in the original resume:")
        for section in missing_sections:
            if section in original_sections:
                specific_instructions.append(f"- Section '{section}' with content like: {original_sections[section][:200]}...")
    
    # Handle experience section issues
    if "Experience" in original_sections:
        original_exp = original_sections["Experience"]
        # Extract company names
        company_pattern = r'\\textbf{([^}]+)}'
        original_companies = re.findall(company_pattern, original_exp)
        
        if "duplicate_jobs" in fix_suggestions or "experience_count" in fix_suggestions:
            specific_instructions.append("\nEXPERIENCE SECTION ISSUE: The Experience section must contain exactly these companies in this order:")
            for i, company in enumerate(original_companies):
                specific_instructions.append(f"  {i+1}. {company}")
    
    # Handle projects section issues
    if "Projects" in original_sections:
        original_proj = original_sections["Projects"]
        # Extract project names
        project_pattern = r'\\textbf{([^}]+)}'
        original_projects = re.findall(project_pattern, original_proj)
        
        if "missing_projects" in fix_suggestions or "projects_structure" in fix_suggestions:
            specific_instructions.append("\nPROJECTS SECTION ISSUE: The Projects section must contain exactly these projects in this order:")
            for i, project in enumerate(original_projects):
                specific_instructions.append(f"  {i+1}. {project}")
    
    # Create the full prompt
    prompt = f"""# RESUME FORMAT CORRECTION TASK

The tailored resume has the following issues that need to be fixed:
{chr(10).join(['- ' + issue for issue in issues])}

## SPECIFIC ISSUES TO CORRECT:
{chr(10).join(specific_instructions)}

## CORRECTION INSTRUCTIONS:
1. Start with the tailored resume provided below
2. Fix ONLY the specific issues mentioned above
3. Keep all the tailored content that doesn't need fixing
4. Maintain the exact LaTeX formatting
5. Return the COMPLETE fixed LaTeX document

## Original Resume:
```latex
{original_latex}

## Tailored Resume with Issues:

{tailored_latex}
Return ONLY the fixed LaTeX content with no additional text before or after.
"""
    return prompt 

    
# ------------------ Resume Tailoring with LLM ------------------
def generate_tailored_resume(resume_text: str, job_description: str, analysis: Dict, relevant_chunks: Optional[List[Document]] = None, model: str = None) -> Dict:
    """Generate a tailored resume using Ollama with robust error handling."""
    # Generate the prompt
    prompt = generate_tailoring_prompt(resume_text, job_description, analysis, relevant_chunks)
    
    try:
        # Call Ollama API
        logger.info(f"Calling Ollama for resume tailoring with model: {model}")
        start_time = time.time()
        
        response = ollama.generate(
            model=model,
            prompt=prompt,
            options={
                "temperature": config.LLM_TEMPERATURE,
                "top_p": 0.9,
                "top_k": 40,
                "num_predict": 4096,
            }
        )
        
        output = response.get('response', '')
        processing_time = time.time() - start_time
        logger.info(f"LLM response received in {processing_time:.2f} seconds")
        
        # Extract the tailored resume and changes
        resume_pattern = r'<TAILORED_RESUME>(.*?)</TAILORED_RESUME>'
        changes_pattern = r'<CHANGES_MADE>(.*?)</CHANGES_MADE>'
        
        resume_match = re.search(resume_pattern, output, re.DOTALL)
        changes_match = re.search(changes_pattern, output, re.DOTALL)
        
        tailored_resume = resume_match.group(1).strip() if resume_match else ""
        changes = changes_match.group(1).strip() if changes_match else ""
        
        logger.info(f"Extracted tailored resume ({len(tailored_resume)} chars) and changes ({len(changes)} chars)")
        
        # NEW: Check if experience section is intact and valid
        experience_check = check_for_experience_integrity(tailored_resume, resume_text)
        if not experience_check["is_valid"]:
            logger.warning(f"Experience section has issues: {', '.join(experience_check['issues'])}")
            # Do a full restoration of the original experience section
            tailored_resume = restore_original_experiences(tailored_resume, resume_text)
            logger.info("Restored original experience section to preserve integrity")
            
            # Add a note about the restoration to the changes
            changes += "\n\nNOTE: The original experience section was preserved to maintain accuracy."
        
        # Validate extracted resume
        if not tailored_resume or "\\begin{document}" not in tailored_resume or "\\end{document}" not in tailored_resume:
            logger.error("Extracted resume is incomplete or invalid")
            
            # Add better error handling to create a valid document if extraction failed
            if not tailored_resume:
                logger.error("No resume content extracted from LLM response")
                tailored_resume = create_fallback_resume(resume_text, job_description, analysis)
            elif "\\begin{document}" not in tailored_resume:
                logger.error("Resume missing \\begin{document}")
                tailored_resume = f"\\documentclass{{article}}\n\\usepackage{{geometry}}\n\\geometry{{margin=1in}}\n\\usepackage{{hyperref}}\n\\begin{{document}}\n{tailored_resume}\n\\end{{document}}"
            elif "\\end{document}" not in tailored_resume:
                logger.error("Resume missing \\end{document}")
                tailored_resume = f"{tailored_resume}\n\\end{{document}}"
        
        # Check document body content
        begin_idx = tailored_resume.find("\\begin{document}")
        end_idx = tailored_resume.find("\\end{document}")
        
        if begin_idx != -1 and end_idx != -1:
            document_body = tailored_resume[begin_idx + len("\\begin{document}"):end_idx].strip()
            if not document_body or len(document_body) < 100:
                logger.error(f"Document body content is minimal: {len(document_body)} chars")
                # Replace with a fallback resume with actual content
                tailored_resume = create_fallback_resume(resume_text, job_description, analysis)
        
        # Auto-wrap any remaining links
        tailored_resume = auto_wrap_links(tailored_resume)
        
        # Fix common LaTeX formatting issues
        tailored_resume = fix_latex_formatting(tailored_resume)
        
        return {
            "status": "success",
            "tailored_resume": tailored_resume,
            "changes": changes,
            "processing_time": processing_time
        }
    except Exception as e:
        logger.exception(f"Error in LLM generation: {str(e)}")
        # Create a fallback document with meaningful content
        fallback_latex = create_fallback_resume(resume_text, job_description, analysis)
        
        return {
            "status": "error",
            "error": str(e),
            "tailored_resume": fallback_latex,
            "changes": f"Error occurred during resume generation: {str(e)}. A fallback resume has been created."
        }

def create_fallback_resume(resume_text: str, job_description: str, analysis: Dict) -> str:
    """
    Create a fallback resume when LLM generation fails.
    Enhanced to better preserve the original LaTeX structure, especially projects and experiences.
    """
    logger.info("Creating fallback resume with preserved original format")
    
    # If we have the original resume in LaTeX format, use its structure
    if resume_text and "\\begin{document}" in resume_text and "\\end{document}" in resume_text:
        try:
            # Extract the full LaTeX document structure
            document_class_match = re.search(r'(\\documentclass.*?)\\begin{document}', resume_text, re.DOTALL)
            document_preamble = document_class_match.group(1) if document_class_match else "\\documentclass{article}\n\\usepackage{geometry}\n\\usepackage{hyperref}\n\\geometry{margin=1in}\n\\hypersetup{colorlinks=true, linkcolor=blue, urlcolor=blue}\n"
            
            # Extract header (everything between \begin{document} and first \section*)
            header_match = re.search(r'\\begin{document}(.*?)\\section\*{', resume_text, re.DOTALL)
            header = header_match.group(1) if header_match else ""
            
            # Extract all sections to preserve their order and format exactly
            sections = {}
            section_pattern = r'\\section\*{([^}]+)}(.*?)(?=\\section\*{|\\end{document}|$)'
            for title, content in re.findall(section_pattern, resume_text, re.DOTALL):
                sections[title] = content.strip()
            
            # Start building the new document
            latex = document_preamble + "\\begin{document}\n\n"
            latex += header
            
            # Process each section in original order
            original_section_order = []
            for title, _ in re.findall(section_pattern, resume_text, re.DOTALL):
                original_section_order.append(title)
            
            for section_title in original_section_order:
                section_content = sections[section_title]
                latex += f"\\section*{{{section_title}}}\n"
                
                # Special handling for specific sections
                if section_title == "Objective":
                    # For Objective, provide a tailored statement
                    match_percentage = analysis.get('match_percentage', 75)  # Default to 75% if not provided
                    
                    # Add more keywords from missing_skills if available
                    additional_keywords = ""
                    if analysis.get("missing_skills"):
                        top_missing = analysis["missing_skills"][:3]  # Use top 3 missing skills
                        if top_missing:
                            additional_keywords = f" Proficient in {', '.join(top_missing)}."
                    
                    latex += f"Passionate and innovative professional seeking opportunities that align with my expertise in {', '.join(analysis.get('matching_skills', ['technology'])[:5])}."
                    latex += f"{additional_keywords} Looking to leverage my skills and experience with a profile that demonstrates a {match_percentage}\\% match with the position requirements.\n\n"
                
                elif section_title == "Experience":
                    # Preserve all original experience entries exactly
                    latex += section_content + "\n\n"
                
                elif section_title == "Projects":
                    # Preserve all original project entries exactly
                    latex += section_content + "\n\n"
                
                elif section_title == "Technical Skills":
                    # Enhance skills section with analysis data
                    if "\\textbf{" in section_content:
                        # If the original format has categories, preserve them
                        latex += section_content + "\n\n"
                    else:
                        # Format skills by category using analysis data
                        skills_by_category = {}
                        
                        # Add matching skills first
                        for skill in analysis.get("matching_skills", []):
                            if "programming" in skill.lower() or "python" in skill.lower() or "c++" in skill.lower():
                                category = "Programming"
                            elif "machine" in skill.lower() or "learning" in skill.lower() or "ai" in skill.lower():
                                category = "Machine Learning & AI"
                            elif "vision" in skill.lower() or "detection" in skill.lower():
                                category = "Computer Vision"
                            else:
                                category = "Other Skills"
                            
                            if category not in skills_by_category:
                                skills_by_category[category] = []
                            skills_by_category[category].append(skill)
                        
                        # Add skills by category
                        for category, skills in skills_by_category.items():
                            if skills:
                                latex += f"\\textbf{{{category}:}} {', '.join([sanitize_input(s) for s in skills])}\n\n"
                        
                        # If no skills were added, use the original content
                        if not skills_by_category:
                            latex += section_content + "\n\n"
                
                elif section_title == "Education":
                    # For Education, either use analysis or original content
                    if analysis.get("education") and "\\begin{itemize}" in section_content:
                        # Preserve the itemize structure but enhance with analysis data
                        latex += "\\begin{itemize}\n"
                        for edu in analysis.get("education", []):
                            institution = sanitize_input(edu.get("institution", ""))
                            degree = sanitize_input(edu.get("degree", ""))
                            year = sanitize_input(edu.get("year", ""))
                            latex += f"    \\item \\textbf{{{institution}}} \\hfill {year}\\newline\n"
                            latex += f"    {degree}\n"
                        latex += "\\end{itemize}\n\n"
                    else:
                        # Just use original content
                        latex += section_content + "\n\n"
                else:
                    # For all other sections, keep the original content
                    latex += section_content + "\n\n"
            
            # Close the document
            latex += "\\end{document}"
            
            return latex
            
        except Exception as e:
            logger.exception(f"Error creating formatted fallback resume: {str(e)}")
            # Fall back to the simple version below if extraction fails
    
    # If we don't have the original LaTeX or extraction failed, create a simple fallback
    logger.info("Creating simple fallback resume (format extraction failed)")
    
    # Extract key information from analysis
    matching_skills = analysis.get("matching_skills", [])
    missing_skills = analysis.get("missing_skills", [])
    education = analysis.get("education", [])
    experience = analysis.get("experience", [])
    responsibilities = analysis.get("responsibilities", [])
    
    # Create a basic formatted resume
    latex = r"\documentclass{article}" + "\n"
    latex += r"\usepackage{geometry}" + "\n"
    latex += r"\usepackage{hyperref}" + "\n"
    latex += r"\geometry{margin=1in}" + "\n"
    latex += r"\hypersetup{colorlinks=true, linkcolor=blue, urlcolor=blue}" + "\n"
    latex += r"\begin{document}" + "\n\n"
    
    # Contact Information
    latex += r"\begin{center}" + "\n"
    latex += r"{\Large \textbf{Pavan Kumar Reddy Kunchala}}" + "\n\n"
    latex += r"Email: Pavankunchalapk@gmail.com" + "\n"
    latex += r"LinkedIn: pavan-kumar-reddy-kunchala — GitHub: Pavankunchala" + "\n"
    latex += r"Portfolio: pavankunchalapk.wixsite.com/resume — Phone: +1 909 402 5512" + "\n"
    latex += r"\end{center}" + "\n\n"
    
    # Objective Section
    latex += r"\section*{Objective}" + "\n"
    
    # Create a better objective by including matching and missing skills
    objective_skills = ", ".join(matching_skills[:5]) if matching_skills else "computer vision and AI technologies"
    missing_skills_text = ""
    if missing_skills:
        top_missing = missing_skills[:3]
        missing_skills_text = f" with knowledge of {', '.join(top_missing)},"
    
    latex += f"Passionate and innovative professional with expertise in {objective_skills}{missing_skills_text} "
    latex += f"seeking opportunities that match my profile which demonstrates a {analysis.get('match_percentage', 75)}\\% match with the position requirements.\n\n"
    
    # Education Section
    latex += r"\section*{Education}" + "\n"
    if education:
        latex += r"\begin{itemize}" + "\n"
        for edu in education:
            institution = sanitize_input(edu.get("institution", ""))
            degree = sanitize_input(edu.get("degree", ""))
            year = sanitize_input(edu.get("year", ""))
            latex += f"    \\item \\textbf{{{institution}}} \\hfill {year}\\newline\n"
            latex += f"    {degree}\n"
        latex += r"\end{itemize}" + "\n\n"
    else:
        latex += "California State University, San Bernardino, USA \\hfill Aug 2023 – Present\n"
        latex += "Master of Science in Computer Science\n\n"
        latex += "Lovely Professional University, India \\hfill July 2018 – Apr 2022\n"
        latex += "Bachelor of Technology in Computer Science\n\n"
    
    # Technical Skills Section
    latex += r"\section*{Technical Skills}" + "\n"
    
    # Categorize skills
    if matching_skills:
        programming_skills = [s for s in matching_skills if "programming" in s.lower() or "python" in s.lower() or "c++" in s.lower()]
        ml_skills = [s for s in matching_skills if "machine" in s.lower() or "learning" in s.lower() or "ai" in s.lower()]
        cv_skills = [s for s in matching_skills if "vision" in s.lower() or "detection" in s.lower()]
        other_skills = [s for s in matching_skills if s not in programming_skills and s not in ml_skills and s not in cv_skills]
        
        if programming_skills:
            latex += f"\\textbf{{Programming:}} {', '.join([sanitize_input(s) for s in programming_skills])}\n\n"
        else:
            latex += "\\textbf{Programming:} Python, C++, C#, HTML\n\n"
            
        if ml_skills:
            latex += f"\\textbf{{Machine Learning & AI:}} {', '.join([sanitize_input(s) for s in ml_skills])}\n\n"
        else:
            latex += "\\textbf{Machine Learning & AI:} Deep Learning, Generative AI, LLMs, RAG, Fine-Tuning\n\n"
            
        if cv_skills:
            latex += f"\\textbf{{Computer Vision:}} {', '.join([sanitize_input(s) for s in cv_skills])}\n\n"
        else:
            latex += "\\textbf{Computer Vision:} OpenCV, PyTorch, TensorFlow-Lite, Kornia, Mediapipe\n\n"
            
        if other_skills:
            latex += f"\\textbf{{Other Skills:}} {', '.join([sanitize_input(s) for s in other_skills])}\n\n"
    else:
        latex += "\\textbf{Programming:} Python, C++, C#, HTML\n\n"
        latex += "\\textbf{Machine Learning & AI:} Deep Learning, Generative AI, LLMs, RAG, Fine-Tuning\n\n"
        latex += "\\textbf{Computer Vision:} OpenCV, PyTorch, TensorFlow-Lite, Kornia, Mediapipe\n\n"
        latex += "\\textbf{Development & Tools:} Linux, Git, GPUs, Unity 3D, Transformers, LangChain\n\n"
    
    # Experience Section
    latex += r"\section*{Experience}" + "\n"
    if experience:
        latex += r"\begin{itemize}" + "\n"
        for exp in experience:
            company = sanitize_input(exp.get("company", ""))
            title = sanitize_input(exp.get("title", ""))
            dates = sanitize_input(exp.get("dates", ""))
            latex += f"    \\item \\textbf{{{company}}} \\hfill {dates}\\newline\n"
            latex += f"    {title}\n"
            latex += "    \\begin{itemize}\n"
            # Add relevant job responsibilities
            added_responsibilities = 0
            for resp in responsibilities:
                if added_responsibilities >= 2:
                    break
                # Check if this responsibility might be relevant to this job
                if any(keyword in resp.lower() for keyword in [company.lower(), title.lower()]):
                    latex += f"        \\item {sanitize_input(resp)}\n"
                    added_responsibilities += 1
            
            # If no specific responsibilities were added, add generic ones
            if added_responsibilities == 0:
                for resp in responsibilities[:2]:
                    latex += f"        \\item {sanitize_input(resp)}\n"
                    
            latex += "    \\end{itemize}\n"
        latex += r"\end{itemize}" + "\n\n"
    else:
        # Add default experience entries
        latex += r"\begin{itemize}" + "\n"
        latex += "    \\item \\textbf{TCCentral (HeyStack)} \\hfill Feb 2024 – Present\\newline\n"
        latex += "    Computer Vision Engineer\n"
        latex += "    \\begin{itemize}\n"
        latex += "        \\item Developed AI-powered models for analysis, enhancing classification and detection capabilities.\n"
        latex += "        \\item Managed workflows and optimized computer vision models for robust real-world performance.\n"
        latex += "    \\end{itemize}\n"
        
        latex += "    \\item \\textbf{Berkeley Synthetic} \\hfill Jan 2023 – Aug 2023\\newline\n"
        latex += "    Generative AI Researcher\n"
        latex += "    \\begin{itemize}\n"
        latex += "        \\item Designed and implemented AI animations using Stable Diffusion and ControlNet.\n"
        latex += "        \\item Researched Text-to-3D models using NeRF-based architectures for advanced reconstruction.\n"
        latex += "    \\end{itemize}\n"
        
        latex += r"\end{itemize}" + "\n\n"
    
    # Projects Section
    latex += r"\section*{Projects}" + "\n"
    latex += r"\begin{itemize}" + "\n"
    
    # Add Law Compass project
    latex += "    \\item \\textbf{Law Compass}\\newline\n"
    latex += "    Developed an end-to-end legal case management platform featuring a state-of-the-art Retrieval-Augmented Generation (RAG) system for generating detailed responses with source citations.\\newline\n"
    latex += "    Website: lawcompass.info\n\n"
    
    # Add CSV File Analyzer project
    latex += "    \\item \\textbf{CSV File Analyzer with CrewAI}\\newline\n"
    latex += "    Automated dataset analysis, cleaning, and visualization using AI-powered LLMs to streamline data-driven decision-making.\n\n"
    
    # Add Medical and Coding Chatbots project
    latex += "    \\item \\textbf{Medical and Coding Chatbots}\\newline\n"
    latex += "    Developed fine-tuned AI chatbots for specialized medical and coding assistance using DSPY and Llama-Index, implementing Grounded RAG solutions for improved accuracy.\n\n"
    
    latex += r"\end{itemize}" + "\n\n"
    
    latex += r"\end{document}"
    
    return latex

def fix_latex_formatting(latex_content: str) -> str:
    """Fix common LaTeX formatting issues."""
    if not latex_content:
        return ""
    
    # Fix nested href commands
    nested_href_pattern = r'\\href{\\href{([^{}]+)}{([^{}]+)}}{([^{}]+)}'
    latex_content = re.sub(nested_href_pattern, r'\\href{\1}{\3}', latex_content)
    
    # Fix double href commands
    double_href_pattern = r'\\href{([^{}]+)}{([^{}]+)}\\href{([^{}]+)}{([^{}]+)}'
    latex_content = re.sub(double_href_pattern, r'\\href{\1}{\2} \\href{\3}{\4}', latex_content)
    
    # Convert hyphenated lists to proper itemize environments
    hyphen_list_pattern = r'(-\s+[^\n]+\n)+'
    
    def replace_hyphen_list(match):
        items = re.findall(r'-\s+([^\n]+)', match.group(0))
        result = "\\begin{itemize}\n"
        for item in items:
            result += f"    \\item {item}\n"
        result += "\\end{itemize}\n"
        return result
    
    latex_content = re.sub(hyphen_list_pattern, replace_hyphen_list, latex_content)
    
    # Ensure proper spacing after sections
    latex_content = re.sub(r'(\\section\*{[^}]+})\n([^\n])', r'\1\n\n\2', latex_content)
    
    # Ensure at least one newline before \end{document}
    if "\\end{document}" in latex_content and not re.search(r'\n\s*\\end{document}', latex_content):
        latex_content = latex_content.replace("\\end{document}", "\n\\end{document}")
    
    return latex_content

# ------------------ PDF Generation ------------------
def compile_latex_to_pdf(latex_content: str) -> Tuple[str, str]:
    """Compiles LaTeX content to a PDF and returns a base64-encoded PDF along with the method used."""
    if not latex_content:
        raise ValueError("No LaTeX content provided")
    
    # Ensure the document has actual content
    begin_idx = latex_content.find("\\begin{document}")
    end_idx = latex_content.find("\\end{document}")
    
    if begin_idx != -1 and end_idx != -1:
        document_body = latex_content[begin_idx + len("\\begin{document}"):end_idx].strip()
        if not document_body or len(document_body) < 50:
            logger.warning(f"Document body is very short ({len(document_body)} chars), padding with minimal content")
            
            # Insert minimal content if document body is empty
            body_insert = "\n\\section*{Resume Content}\nThis resume has been automatically generated.\n\n"
            latex_content = latex_content[:begin_idx + len("\\begin{document}")] + body_insert + latex_content[end_idx:]
    
    for generator in pdf_generators:
        if not generator.available:
            continue
        try:
            logger.info(f"Attempting PDF generation using {generator.name}")
            if generator.name == "pdflatex":
                return compile_with_pdflatex(latex_content), generator.name
            elif generator.name == "reportlab":
                return compile_with_reportlab(latex_content), generator.name
            elif generator.name == "error_pdf":
                return generate_error_pdf(), generator.name
        except Exception as e:
            logger.exception(f"{generator.name} failed: {str(e)}")
    
    raise RuntimeError("All PDF generation methods failed")

def compile_with_pdflatex(latex_content: str) -> str:
    """Uses pdflatex to compile LaTeX and returns a base64-encoded PDF."""
    with temp_file_manager(suffix='.tex') as temp_tex_path:
        temp_tex_path.write_text(latex_content, encoding='utf-8')
        temp_dir = str(temp_tex_path.parent)
        base_name = temp_tex_path.stem
        
        # Run pdflatex with better error handling
        try:
            process = subprocess.Popen(
                ['pdflatex', '-interaction=nonstopmode', '-output-directory', temp_dir, str(temp_tex_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            stdout, stderr = process.communicate(timeout=30)  # Add timeout
            
            if process.returncode != 0:
                logger.error(f"pdflatex error: {stderr}")
                # Check for common LaTeX errors
                if "Emergency stop" in stdout:
                    latex_err = re.search(r'! (.+?)\.', stdout)
                    error_msg = latex_err.group(1) if latex_err else "Unknown LaTeX error"
                    raise RuntimeError(f"LaTeX compilation failed: {error_msg}")
                else:
                    raise RuntimeError("Failed to compile LaTeX with pdflatex")
            
            pdf_path = Path(temp_dir) / f"{base_name}.pdf"
            if not pdf_path.exists():
                logger.error("PDF file was not created despite successful return code")
                raise FileNotFoundError("PDF file was not created by pdflatex")
            
            pdf_base64 = base64.b64encode(pdf_path.read_bytes()).decode('utf-8')
            
            # Cleanup auxiliary files
            for ext in ['.aux', '.log', '.out']:
                aux_file = pdf_path.with_suffix(ext)
                if aux_file.exists():
                    aux_file.unlink()
            
            return pdf_base64
        except subprocess.TimeoutExpired:
            logger.error("pdflatex process timed out")
            raise RuntimeError("LaTeX compilation timed out")

def extract_text_from_latex(latex_content: str) -> str:
    """Extracts plain text from LaTeX for fallback PDF generation."""
    # Remove comments
    text = re.sub(r'%.*$', '', latex_content, flags=re.MULTILINE)
    
    # Extract main document body
    doc_match = re.search(r'\\begin{document}(.*?)\\end{document}', text, re.DOTALL)
    if doc_match:
        text = doc_match.group(1)
    
    # Remove common LaTeX commands
    text = re.sub(r'\\(section|subsection|textbf|textit|emph){([^}]*)}', r'\2', text)
    text = re.sub(r'\\[a-zA-Z]+(\[[^\]]*\])?{([^}]*)}', r'\2', text)
    text = re.sub(r'\\[a-zA-Z]+', ' ', text)
    
    # Replace escaped characters
    text = text.replace(r'\&', '&').replace(r'\%', '%').replace(r'\_', '_')
    text = re.sub(r'\\begin{[^}]*}|\\end{[^}]*}', '', text)
    
    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def compile_with_reportlab(latex_content: str) -> str:
    """Uses ReportLab to generate a PDF from plain text extracted from LaTeX."""
    try:
        text_content = extract_text_from_latex(latex_content)
        buffer = io.BytesIO()
        
        # Create the PDF document
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        story = [Paragraph("Resume", styles['Title']), Spacer(1, 12)]
        
        # Style for code blocks
        code_style = ParagraphStyle(
            'Code', 
            parent=styles['Normal'],
            fontName='Courier',
            fontSize=9,
            leading=11
        )
        
        # Extract sections if possible
        section_pattern = r'\\section\*{([^}]+)}(.*?)(?=\\section\*{|\\end{document}|$)'
        sections = re.findall(section_pattern, latex_content, re.DOTALL)
        
        if sections:
            # Process each section
            for section_title, section_content in sections:
                # Add section title
                story.append(Paragraph(section_title, styles['Heading2']))
                story.append(Spacer(1, 6))
                
                # Clean section content
                clean_content = extract_text_from_latex(section_content)
                
                # Split into paragraphs
                paragraphs = [p.strip() for p in clean_content.split('\n\n') if p.strip()]
                
                # Add each paragraph
                for para in paragraphs:
                    story.append(Paragraph(para, styles['Normal']))
                    story.append(Spacer(1, 6))
                
                story.append(Spacer(1, 12))
        else:
            # Add content paragraphs if no sections found
            for para in text_content.split('\n\n'):
                if para.strip():
                    style = code_style if para.startswith(("\\", "%")) else styles['Normal']
                    story.append(Paragraph(para, style))
                    story.append(Spacer(1, 6))
        
        # Build the PDF
        doc.build(story)
        pdf_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return pdf_base64
    except Exception as e:
        logger.exception(f"ReportLab PDF generation failed: {str(e)}")
        raise

def generate_error_pdf() -> str:
    """Generates a simple error PDF indicating failure in PDF generation."""
    try:
        buffer = io.BytesIO()
        
        # Create a simple canvas
        c = canvas.Canvas(buffer, pagesize=letter)
        width, height = letter
        
        # Add error message
        c.setFont("Helvetica-Bold", 16)
        c.drawString(100, height - 100, "PDF Generation Error")
        
        c.setFont("Helvetica", 12)
        c.drawString(100, height - 150, "Failed to generate PDF from LaTeX content.")
        c.drawString(100, height - 170, "Please use the LaTeX code directly.")
        
        c.save()
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    except Exception as e:
        logger.exception(f"Error PDF generation failed: {str(e)}")
        raise RuntimeError("Failed to generate error PDF")
    

def restore_original_experiences(tailored_latex: str, original_latex: str) -> str:
    """
    Completely replace the experience section in the tailored resume with the original
    experience section, preserving all companies, dates, and positions.
    """
    if not original_latex or "\\section*{Experience}" not in original_latex:
        return tailored_latex
    
    # Extract the original Experience section
    orig_exp_pattern = r'(\\section\*{Experience})(.*?)(?=\\section\*{|\\end{document}|$)'
    orig_match = re.search(orig_exp_pattern, original_latex, re.DOTALL)
    
    if not orig_match:
        return tailored_latex
    
    orig_section_header = orig_match.group(1)
    orig_exp_content = orig_match.group(2)
    
    # Extract individual company entries from original
    company_entries = []
    itemize_pattern = r'\\begin{itemize}(.*?)\\end{itemize}'
    itemize_match = re.search(itemize_pattern, orig_exp_content, re.DOTALL)
    
    if itemize_match:
        itemize_content = itemize_match.group(1)
        entry_pattern = r'(\\item\\textbf{[^}]+}.*?)(?=\\item\\textbf{|$)'
        company_entries = re.findall(entry_pattern, itemize_content, re.DOTALL)
    
    # Nothing to restore if we couldn't extract the original entries
    if not company_entries:
        return tailored_latex
    
    # Now find the Experience section in the tailored resume to replace
    tailored_exp_pattern = r'(\\section\*{Experience})(.*?)(?=\\section\*{|\\end{document}|$)'
    tailored_match = re.search(tailored_exp_pattern, tailored_latex, re.DOTALL)
    
    if not tailored_match:
        # If no Experience section in tailored resume, add it back
        # Find a good position to insert it (before Projects if it exists)
        projects_match = re.search(r'\\section\*{Projects}', tailored_latex)
        if projects_match:
            insert_pos = projects_match.start()
            # Build the Experience section
            exp_section = f"{orig_section_header}{orig_exp_content}\n\n"
            # Insert before Projects
            return tailored_latex[:insert_pos] + exp_section + tailored_latex[insert_pos:]
        else:
            # Add at end of document
            end_doc_pos = tailored_latex.find('\\end{document}')
            if end_doc_pos == -1:
                return tailored_latex
            
            exp_section = f"{orig_section_header}{orig_exp_content}\n\n"
            return tailored_latex[:end_doc_pos] + exp_section + tailored_latex[end_doc_pos:]
    
    # Replace the entire Experience section
    return re.sub(
        tailored_exp_pattern,
        lambda m: f"{orig_section_header}{orig_exp_content}",
        tailored_latex,
        flags=re.DOTALL
    )


def evaluate_resume_content(original_latex: str, tailored_latex: str) -> Dict[str, Any]:
    """
    Thoroughly evaluates the tailored resume for content and formatting issues.
    Enhanced to detect duplicate experience entries and other structural problems.
    """
    issues = []
    metrics = {}
    is_valid = True
    fix_suggestions = {}
    
    try:
        # Basic validation
        if not tailored_latex or "\\begin{document}" not in tailored_latex or "\\end{document}" not in tailored_latex:
            issues.append("Invalid LaTeX structure: missing document tags")
            is_valid = False
            return {
                "is_valid": is_valid,
                "issues": issues,
                "metrics": metrics,
                "fix_suggestions": {"general": "Regenerate the entire resume with proper LaTeX structure"}
            }
        
        # Extract sections from both original and tailored resumes
        def extract_sections(latex_content):
            sections = {}
            section_pattern = r'\\section\*{([^}]+)}(.*?)(?=\\section\*{|\\end{document}|$)'
            for title, content in re.findall(section_pattern, latex_content, re.DOTALL):
                sections[title] = content.strip()
            return sections
        
        original_sections = extract_sections(original_latex)
        tailored_sections = extract_sections(tailored_latex)
        
        # Compare section presence
        original_section_names = set(original_sections.keys())
        tailored_section_names = set(tailored_sections.keys())
        
        missing_sections = original_section_names - tailored_section_names
        added_sections = tailored_section_names - original_section_names
        
        if missing_sections:
            issues.append(f"Missing sections: {', '.join(missing_sections)}")
            is_valid = False
            fix_suggestions["missing_sections"] = f"Add the following sections back: {', '.join(missing_sections)}"
        
        if added_sections:
            issues.append(f"Added sections not in original: {', '.join(added_sections)}")
            is_valid = False
            fix_suggestions["added_sections"] = f"Remove these sections: {', '.join(added_sections)}"
        
        # Check for Projects section specifically
        if "Projects" in original_sections and ("Projects" not in tailored_sections or not tailored_sections["Projects"].strip()):
            issues.append("Projects section is missing or empty")
            is_valid = False
            fix_suggestions["projects_missing"] = "Copy the entire Projects section from the original resume"
        
        # Extract and compare company names in Experience section
        if "Experience" in original_sections and "Experience" in tailored_sections:
            # Extract company names from both
            company_pattern = r'\\textbf{([^}]+)}'
            original_companies = re.findall(company_pattern, original_sections["Experience"])
            tailored_companies = re.findall(company_pattern, tailored_sections["Experience"])
            
            # Check for exact count of company mentions
            if len(original_companies) != len(tailored_companies):
                issues.append(f"Experience section has wrong number of entries: Original had {len(original_companies)}, tailored has {len(tailored_companies)}")
                is_valid = False
                fix_suggestions["experience_count"] = f"Ensure the Experience section has EXACTLY {len(original_companies)} entries, no more and no less"
            
            # Check for missing companies
            original_company_set = set(original_companies)
            tailored_company_set = set(tailored_companies)
            
            missing_companies = original_company_set - tailored_company_set
            if missing_companies:
                issues.append(f"Missing companies in Experience section: {', '.join(missing_companies)}")
                is_valid = False
                fix_suggestions["missing_companies"] = f"Add back the following companies: {', '.join(missing_companies)}"
            
            # Check for duplicates - this is especially important given the observed issue
            company_counts = {}
            for company in tailored_companies:
                company_counts[company] = company_counts.get(company, 0) + 1
            
            duplicated_companies = {company: count for company, count in company_counts.items() if count > 1}
            
            if duplicated_companies:
                duplicates_str = ", ".join([f"{company} ({count} times)" for company, count in duplicated_companies.items()])
                issues.append(f"Duplicated job entries: {duplicates_str}")
                is_valid = False
                fix_suggestions["duplicate_jobs"] = "Remove duplicate job entries, keeping only one instance of each company"
                
                # Add more specific instructions to fix the duplicates
                for company, count in duplicated_companies.items():
                    fix_suggestions[f"duplicate_{company}"] = f"Keep only the first instance of '{company}' and remove the other {count-1} duplicate entries"
            
            # Check for order changes
            for i in range(min(len(original_companies), len(tailored_companies))):
                if i < len(original_companies) and i < len(tailored_companies) and original_companies[i] != tailored_companies[i]:
                    issues.append(f"Experience order changed: '{original_companies[i]}' should be at position {i+1}, but found '{tailored_companies[i]}' instead")
                    is_valid = False
                    fix_suggestions["experience_order"] = "Restore the original order of experience entries"
                    break
        
        # Check Project section formatting and content
        if "Projects" in original_sections and "Projects" in tailored_sections:
            original_projects = original_sections["Projects"]
            tailored_projects = tailored_sections["Projects"]
            
            # Extract project names from both
            project_name_pattern = r'\\textbf{([^}]+)}'
            original_project_names = re.findall(project_name_pattern, original_projects)
            tailored_project_names = re.findall(project_name_pattern, tailored_projects)
            
            # Check for missing projects
            original_project_set = set(original_project_names)
            tailored_project_set = set(tailored_project_names)
            
            missing_projects = original_project_set - tailored_project_set
            if missing_projects:
                issues.append(f"Missing projects: {', '.join(missing_projects)}")
                is_valid = False
                fix_suggestions["missing_projects"] = f"Add back the following projects: {', '.join(missing_projects)}"
            
            # Check for added projects
            added_projects = tailored_project_set - original_project_set
            if added_projects:
                issues.append(f"Added projects not in original: {', '.join(added_projects)}")
                is_valid = False
                fix_suggestions["added_projects"] = f"Remove these projects: {', '.join(added_projects)}"
            
            # Check for structure preservation (itemize environments)
            original_itemize_count = original_projects.count("\\begin{itemize}")
            tailored_itemize_count = tailored_projects.count("\\begin{itemize}")
            
            if original_itemize_count != tailored_itemize_count:
                issues.append(f"Projects section structure changed: Original had {original_itemize_count} itemize environments, tailored has {tailored_itemize_count}")
                is_valid = False
                fix_suggestions["projects_structure"] = "Match the exact structure of the original Projects section"
            
            # Check project order
            for i in range(min(len(original_project_names), len(tailored_project_names))):
                if i < len(original_project_names) and i < len(tailored_project_names) and original_project_names[i] != tailored_project_names[i]:
                    issues.append(f"Project order changed: '{original_project_names[i]}' should be at position {i+1}, but found '{tailored_project_names[i]}' instead")
                    is_valid = False
                    fix_suggestions["project_order"] = "Restore the original order of projects"
                    break
        
        # Calculate metrics
        metrics = {
            "original_section_count": len(original_sections),
            "tailored_section_count": len(tailored_sections),
            "shared_section_count": len(original_section_names.intersection(tailored_section_names)),
            "format_preservation_score": 100 - (len(issues) * 10) if is_valid else 0  # Deduct 10 points per issue
        }
        
    except Exception as e:
        issues.append(f"Evaluation error: {str(e)}")
        is_valid = False
        fix_suggestions["general"] = "Regenerate the resume due to evaluation error"
        logger.error(f"Error in resume evaluation: {str(e)}", exc_info=True)
    
    return {
        "is_valid": is_valid,
        "issues": issues,
        "metrics": metrics,
        "fix_suggestions": fix_suggestions
    }

# ------------------ Main Processing Pipeline ------------------
def process_resume_and_job(resume_content: str, job_description: str, model: str, resume_format: str = 'txt') -> Dict:
    """
    Process the resume and job description using the optimized retrieval pipeline.
    Enhanced with format preservation, evaluation, and better error handling.
    
    Args:
        resume_content: The resume content (text or PDF content)
        job_description: The job description text
        model: The Ollama model to use
        resume_format: Format of the resume ('txt' or 'pdf')
    
    Returns:
        Dict containing latex, pdf, and summary information
    """
    start_time = time.time()
    process_id = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
    logger.info(f"[Process-{process_id}] Starting new resume tailoring request with format: {resume_format}")
    
    try:
        # Track original LaTeX for format comparison
        original_latex = None
        
        # Extract text from resume if it's a PDF
        resume_text = None
        if resume_format.lower() == 'pdf':
            logger.info(f"[Process-{process_id}] Extracting text from PDF resume")
            resume_text = extract_text_from_pdf(resume_content)
            if resume_text:
                # Log a sample of extracted text for debugging
                sample = resume_text[:300].replace('\n', ' ')
                logger.info(f"[Process-{process_id}] Extracted resume text sample: {sample}...")
            else:
                logger.error(f"[Process-{process_id}] Failed to extract text from PDF resume")
        else:
            resume_text = resume_content
            # If the resume is already in LaTeX format, store it for comparison
            if isinstance(resume_text, str) and "\\begin{document}" in resume_text:
                original_latex = resume_text
        
        # Validate resume text
        if not resume_text or len(resume_text.strip()) < config.MIN_RESUME_LENGTH:
            logger.error(f"[Process-{process_id}] Resume text is too short ({len(resume_text) if resume_text else 0} chars)")
            return {
                "status": "error",
                "error": "The resume text is too short or could not be extracted properly.",
                "latex": create_fallback_resume("", job_description, {}),
                "pdf": generate_error_pdf(),
                "pdf_method": "error_pdf",
                "summary": "Error: The resume text is too short or could not be extracted properly."
            }
        
        logger.info(f"[Process-{process_id}] Valid resume text extracted ({len(resume_text)} chars)")
        
        # Compute content-based ID for caching and persistence
        collection_id = compute_collection_id(resume_text, job_description)
        logger.info(f"[Process-{process_id}] Using collection ID: {collection_id}")
        
        # Create chunks for both resume and job description
        resume_chunks = create_chunks(resume_text, "resume")
        job_chunks = create_chunks(job_description, "job")
        
        # Log chunk statistics
        logger.info(f"[Process-{process_id}] Created {len(resume_chunks)} resume chunks and {len(job_chunks)} job chunks")
        
        # Create or load vector store if embeddings are available
        vector_store = None
        relevant_chunks = None
        retrieval_method = "none"
        
        if embeddings:
            # Create or load a vector store for the collection
            vector_store = create_or_load_vector_store(resume_chunks + job_chunks, collection_id)
            
            if vector_store:
                logger.info(f"[Process-{process_id}] Vector store ready for collection {collection_id}")
                
                # Create retriever based on configuration
                retriever = create_hybrid_retriever(resume_chunks, job_chunks, vector_store, collection_id)
                
                if retriever:
                    # Determine retrieval method for logging
                    if isinstance(retriever, dict) and 'bm25' in retriever and 'vector' in retriever:
                        retrieval_method = "all"
                    elif config.RETRIEVAL_MODE.lower() == 'hybrid':
                        retrieval_method = "hybrid"
                    elif config.RETRIEVAL_MODE.lower() == 'dense':
                        retrieval_method = "dense"
                    elif config.RETRIEVAL_MODE.lower() == 'bm25':
                        retrieval_method = "bm25"
                    
                    # Perform retrieval with dynamic queries
                    relevant_chunks = retrieve_relevant_chunks(retriever, resume_chunks, job_chunks)
                    
                    if relevant_chunks:
                        logger.info(f"[Process-{process_id}] Retrieved {len(relevant_chunks)} relevant chunks using {retrieval_method} retriever")
                        # Log sample of retrieved chunks
                        if len(relevant_chunks) > 0:
                            resume_samples = [c for c in relevant_chunks[:2] if c.metadata.get("source") == "resume"]
                            if resume_samples:
                                sample_text = resume_samples[0].page_content[:100].replace('\n', ' ')
                                logger.info(f"[Process-{process_id}] Resume chunk sample: {sample_text}...")
                    else:
                        logger.warning(f"[Process-{process_id}] No relevant chunks retrieved from retriever")
                else:
                    logger.warning(f"[Process-{process_id}] Failed to create retriever")
            else:
                logger.warning(f"[Process-{process_id}] Failed to create vector store")
        else:
            logger.info(f"[Process-{process_id}] No embeddings available - will use direct prompt with full texts")
        
        # Use LLM to analyze resume and job description
        try:
            logger.info(f"[Process-{process_id}] Analyzing resume and job description with LLM...")
            analysis = analyze_with_llm(resume_text, job_description, model)
            logger.info(f"[Process-{process_id}] Analysis complete - Match percentage: {analysis.get('match_percentage')}%")
            
            # Log analysis summary
            log_msg = f"[Process-{process_id}] Analysis summary - "
            log_msg += f"Matching skills: {len(analysis.get('matching_skills', []))}, "
            log_msg += f"Missing skills: {len(analysis.get('missing_skills', []))}, "
            log_msg += f"Education entries: {len(analysis.get('education', []))}, "
            log_msg += f"Experience entries: {len(analysis.get('experience', []))}"
            logger.info(log_msg)
        except Exception as e:
            logger.error(f"[Process-{process_id}] Error in LLM analysis: {str(e)}")
            # Create a default analysis
            analysis = {
                "matching_skills": [],
                "missing_skills": [],
                "education": [],
                "experience": [],
                "responsibilities": [],
                "match_percentage": 0,
                "error": str(e)
            }
        
        # Generate tailored resume
        try:
            logger.info(f"[Process-{process_id}] Generating tailored resume...")
            result = generate_tailored_resume(resume_text, job_description, analysis, relevant_chunks, model)
            
            if result.get("status") == "error":
                logger.error(f"[Process-{process_id}] Error in tailoring: {result.get('error')}")
            else:
                logger.info(f"[Process-{process_id}] Resume tailoring completed successfully")
                # Log latex size
                latex_size = len(result.get("tailored_resume", ""))
                logger.info(f"[Process-{process_id}] Generated LaTeX size: {latex_size} chars")
        except Exception as e:
            logger.error(f"[Process-{process_id}] Error in resume tailoring: {str(e)}")
            result = {
                "status": "error",
                "error": str(e),
                "tailored_resume": create_fallback_resume(resume_text, job_description, analysis),
                "changes": f"Error occurred: {str(e)}"
            }
        
        # Generate PDF
        try:
            logger.info(f"[Process-{process_id}] Generating PDF from LaTeX...")
            pdf_base64, pdf_method = compile_latex_to_pdf(result.get("tailored_resume", ""))
            logger.info(f"[Process-{process_id}] PDF generation successful using {pdf_method}")
        except Exception as e:
            logger.error(f"[Process-{process_id}] Error in PDF generation: {str(e)}")
            pdf_base64 = generate_error_pdf()
            pdf_method = "error_pdf"
        
        # Prepare response
        total_time = time.time() - start_time
        logger.info(f"[Process-{process_id}] Total processing time: {total_time:.2f} seconds")
        
        response = {
            "status": "success",
            "latex": result.get("tailored_resume", ""),
            "pdf": pdf_base64,
            "summary": result.get("changes", ""),
            "analysis": analysis,
            "pdf_method": pdf_method,
            "processing_time": total_time,
            "model_used": model,
            "embedding_used": embeddings is not None,
            "retrieval_method": retrieval_method,
            "collection_id": collection_id
        }
        
        # Add original latex for format comparison if available
        if original_latex:
            response["original_latex"] = original_latex
        
        return response
    except Exception as e:
        logger.exception(f"[Process-{process_id}] Error in main processing pipeline: {str(e)}")
        
        # Create a simple fallback document
        fallback_latex = (
            r"\documentclass{article}" "\n" +
            r"\usepackage{geometry}" "\n" +
            r"\usepackage{hyperref}" "\n" +
            r"\geometry{margin=1in}" "\n" +
            r"\begin{document}" "\n" +
            r"\section*{Error Processing Resume}" "\n" +
            f"An error occurred: {str(e)}" "\n\n" +
            r"Original Resume:" "\n" +
            sanitize_input(resume_content[:500] if isinstance(resume_content, str) else "Cannot display resume content") + "\n" +
            r"\end{document}"
        )
        
        try:
            pdf_base64, pdf_method = compile_latex_to_pdf(fallback_latex)
        except Exception:
            pdf_base64 = generate_error_pdf()
            pdf_method = "error_pdf"
        
        return {
            "status": "error",
            "latex": fallback_latex,
            "pdf": pdf_base64,
            "summary": f"Error: {str(e)}",
            "analysis": {"error": str(e)},
            "pdf_method": pdf_method,
            "error": str(e),
            "model_used": model
        }

# ------------------ Flask API Endpoints ------------------
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint with enhanced system status information."""
    try:
        # Test the embeddings to ensure they're working
        embeddings_status = "not_available"
        vector_size = 0
        embedding_test_results = {}
        
        if embeddings:
            try:
                # Test with multiple phrases
                test_queries = ["test query", "resume skills", "job requirements"]
                test_results = []
                
                for query in test_queries:
                    test_vector = embeddings.embed_query(query)
                    test_results.append({
                        "query": query,
                        "vector_size": len(test_vector),
                        "sample_values": test_vector[:3]  # Include a few sample values
                    })
                
                vector_size = test_results[0]["vector_size"] if test_results else 0
                embeddings_status = "working"
                embedding_test_results = {
                    "test_count": len(test_results),
                    "test_samples": test_results
                }
            except Exception as e:
                embeddings_status = f"error: {str(e)}"
        
        # Get cache stats
        cache_stats = {
            "vector_store_cache_size": len(vector_store_cache),
            "retriever_cache_size": len(retriever_cache),
            "document_hash_cache_size": len(document_hash_cache)
        }
        
        # Get PDF generator info
        pdf_generator_info = []
        for gen in pdf_generators:
            if gen.available:
                pdf_generator_info.append({
                    "name": gen.name,
                    "priority": gen.priority
                })
        
        return jsonify({
            'status': 'ok',
            'gpu_available': config.USE_GPU,
            'embeddings_status': embeddings_status,
            'retrieval_mode': config.RETRIEVAL_MODE,
            'cache_stats': cache_stats,
            'pdf_generators': pdf_generator_info,
            'default_model': config.DEFAULT_MODEL,
            'embeddings_model': config.EMBEDDING_MODEL if embeddings else "none",
            'embedding_vector_size': vector_size,
            'timestamp': time.time(),
            'config': {
                'chunk_size': config.CHUNK_SIZE,
                'chunk_overlap': config.CHUNK_OVERLAP,
                'bm25_weight': config.BM25_WEIGHT,
                'dense_weight': config.DENSE_WEIGHT,
                'min_chunks_per_source': config.MIN_CHUNKS_PER_SOURCE,
                'min_resume_length': config.MIN_RESUME_LENGTH,
                'pdf_retry_count': config.PDF_RETRY_COUNT
            }
        }), 200
    except Exception as e:
        logger.exception(f"Health check error: {str(e)}")
        return jsonify({'status': 'error', 'error': str(e)}), 500

@app.route('/models', methods=['GET'])
def get_models():
    """List available Ollama models."""
    try:
        models_response = ollama.list()
        models = [{'name': model.get('name')} for model in models_response.get('models', [])]
        return jsonify({'models': models}), 200
    except Exception as e:
        logger.exception(f"Error fetching models: {str(e)}")
        return jsonify({'error': str(e), 'models': []}), 500

@app.route('/cache/clear', methods=['POST'])
def clear_cache():
    """Clear the vector store and retriever caches."""
    try:
        vector_store_cache.clear()
        retriever_cache.clear()
        document_hash_cache.clear()
        
        return jsonify({
            'status': 'success',
            'message': 'All caches cleared successfully'
        }), 200
    except Exception as e:
        logger.exception(f"Error clearing caches: {str(e)}")
        return jsonify({'status': 'error', 'error': str(e)}), 500

@app.route('/config', methods=['GET', 'POST'])
def manage_config():
    """Get or update configuration settings."""
    if request.method == 'GET':
        # Return current config
        return jsonify({
            'retrieval_mode': config.RETRIEVAL_MODE,
            'chunk_size': config.CHUNK_SIZE,
            'chunk_overlap': config.CHUNK_OVERLAP,
            'bm25_weight': config.BM25_WEIGHT,
            'dense_weight': config.DENSE_WEIGHT,
            'llm_temperature': config.LLM_TEMPERATURE,
            'min_chunks_per_source': config.MIN_CHUNKS_PER_SOURCE,
            'max_cache_size': config.MAX_CACHE_SIZE,
            'min_resume_length': config.MIN_RESUME_LENGTH,
            'pdf_retry_count': config.PDF_RETRY_COUNT
        }), 200
    elif request.method == 'POST':
        try:
            data = request.json
            if not data:
                return jsonify({'error': 'No JSON data provided'}), 400
            
            # Update supported config values
            if 'retrieval_mode' in data:
                mode = data['retrieval_mode'].lower()
                if mode in ['hybrid', 'dense', 'bm25', 'all']:
                    config.RETRIEVAL_MODE = mode
                else:
                    return jsonify({'error': 'Invalid retrieval_mode'}), 400
            
            if 'bm25_weight' in data:
                weight = float(data['bm25_weight'])
                if 0 <= weight <= 1:
                    config.BM25_WEIGHT = weight
                    config.DENSE_WEIGHT = 1.0 - weight
                else:
                    return jsonify({'error': 'bm25_weight must be between 0 and 1'}), 400
            
            if 'llm_temperature' in data:
                temp = float(data['llm_temperature'])
                if 0 <= temp <= 1:
                    config.LLM_TEMPERATURE = temp
                else:
                    return jsonify({'error': 'llm_temperature must be between 0 and 1'}), 400
            
            if 'min_chunks_per_source' in data:
                min_chunks = int(data['min_chunks_per_source'])
                if min_chunks >= 0:
                    config.MIN_CHUNKS_PER_SOURCE = min_chunks
                else:
                    return jsonify({'error': 'min_chunks_per_source must be non-negative'}), 400
                    
            if 'min_resume_length' in data:
                min_length = int(data['min_resume_length'])
                if min_length >= 0:
                    config.MIN_RESUME_LENGTH = min_length
                else:
                    return jsonify({'error': 'min_resume_length must be non-negative'}), 400
                    
            if 'pdf_retry_count' in data:
                retry_count = int(data['pdf_retry_count'])
                if retry_count > 0:
                    config.PDF_RETRY_COUNT = retry_count
                else:
                    return jsonify({'error': 'pdf_retry_count must be positive'}), 400
            
            # After updating config, clear caches to ensure consistent behavior
            vector_store_cache.clear()
            retriever_cache.clear()
            
            return jsonify({
                'status': 'success',
                'message': 'Configuration updated',
                'current_config': {
                    'retrieval_mode': config.RETRIEVAL_MODE,
                    'bm25_weight': config.BM25_WEIGHT,
                    'dense_weight': config.DENSE_WEIGHT,
                    'llm_temperature': config.LLM_TEMPERATURE,
                    'min_chunks_per_source': config.MIN_CHUNKS_PER_SOURCE,
                    'min_resume_length': config.MIN_RESUME_LENGTH,
                    'pdf_retry_count': config.PDF_RETRY_COUNT
                }
            }), 200
        except Exception as e:
            logger.exception(f"Error updating config: {str(e)}")
            return jsonify({'status': 'error', 'error': str(e)}), 500

@app.route('/process', methods=['POST'])
def process_endpoint():
    """Process the resume and job description with enhanced validation and error correction."""
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        # Extract request parameters
        resume = data.get('resume', '')
        job_description = data.get('jobDescription', '')
        model = data.get('model', config.DEFAULT_MODEL)
        resume_format = data.get('resumeFormat', 'txt')
        
        # Track the original resume text for evaluation
        original_resume_latex = None
        
        # If resume is a base64 data URL, ensure it's treated as PDF
        if isinstance(resume, str) and resume.startswith('data:application/pdf;base64,'):
            resume_format = 'pdf'
            logger.info(f"Detected base64 PDF data URL ({len(resume)} chars)")
        
        # Validate inputs
        if not resume:
            return jsonify({'error': 'Resume is required'}), 400
        if not job_description:
            return jsonify({'error': 'Job description is required'}), 400
        
        # Log input sizes
        input_size = len(resume) if isinstance(resume, str) else "binary data"
        logger.info(f"Processing resume ({input_size}) and job description ({len(job_description)} chars)")
        logger.info(f"Resume format: {resume_format}")
        
        # Validate model
        try:
            models_response = ollama.list()
            available_models = [model.get('name') for model in models_response.get('models', [])]
            
            if model not in available_models:
                logger.warning(f"Requested model {model} not available, falling back to {config.DEFAULT_MODEL}")
                model = config.DEFAULT_MODEL
        except Exception as e:
            logger.error(f"Error validating model: {str(e)}")
            # Continue with requested model
        
        # Process the resume and job description
        result = process_resume_and_job(resume, job_description, model, resume_format)
        
        # If processing completed, get the original resume LaTeX for evaluation
        if result.get("status") == "success" and result.get("original_latex"):
            original_resume_latex = result.get("original_latex")
        elif resume_format != 'pdf' and isinstance(resume, str) and "\\begin{document}" in resume:
            # If the original resume was submitted as LaTeX, use it directly
            original_resume_latex = resume
        
        # Perform enhanced evaluation if we have both the original and tailored LaTeX
        if original_resume_latex and result.get("tailored_resume"):
            tailored_latex = result.get("tailored_resume")
            
            # Evaluate the tailored resume
            evaluation = evaluate_resume_content(original_resume_latex, tailored_latex)
            result["evaluation"] = evaluation
            
            # Add evaluation summary to the result
            if not evaluation["is_valid"]:
                # Always attempt correction for any issues found, regardless of model
                logger.warning(f"Format issues detected: {len(evaluation['issues'])} issues")
                
                # Create a targeted correction prompt
                fix_prompt = create_section_correction_prompt(original_resume_latex, tailored_latex, evaluation)
                
                try:
                    # Call Ollama API for correction
                    fix_response = ollama.generate(
                        model=model,
                        prompt=fix_prompt,
                        options={
                            "temperature": 0.1,  # Keep temperature low for fixing
                            "num_predict": 8192,  # Use larger token limit for complete resume
                        }
                    )
                    
                    fixed_latex = fix_response.get('response', '')
                    
                    # Validate if it contains proper LaTeX
                    if "\\begin{document}" in fixed_latex and "\\end{document}" in fixed_latex:
                        # Re-evaluate the fixed version
                        re_evaluation = evaluate_resume_content(original_resume_latex, fixed_latex)
                        
                        if re_evaluation["is_valid"] or len(re_evaluation["issues"]) < len(evaluation["issues"]):
                            # Use the improved version
                            result["tailored_resume"] = fixed_latex
                            result["evaluation"] = re_evaluation
                            result["summary"] += "\n\nAutomatic format corrections were applied."
                            logger.info("Successfully applied automatic format corrections")
                            
                            # Try to regenerate PDF with fixed version
                            try:
                                pdf_base64, pdf_method = compile_latex_to_pdf(fixed_latex)
                                result["pdf"] = pdf_base64
                                result["pdf_method"] = pdf_method
                                logger.info("Successfully regenerated PDF after corrections")
                            except Exception as pdf_error:
                                logger.error(f"Error regenerating PDF after fix: {str(pdf_error)}")
                        else:
                            logger.warning("Fix attempt did not improve the resume format")
                            # Try a fallback approach for persistent issues
                            if "Projects" in evaluation["issues"][0] or "Experience" in evaluation["issues"][0]:
                                logger.info("Attempting section-specific fallback for Projects/Experience")
                                try:
                                    # Create a special fallback that preserves all sections but with special attention
                                    # to Projects and Experience sections from the original resume
                                    fallback_latex = create_fallback_resume(original_resume_latex, job_description, analysis=result.get("analysis", {}))
                                    fallback_evaluation = evaluate_resume_content(original_resume_latex, fallback_latex)
                                    
                                    if fallback_evaluation["is_valid"] or len(fallback_evaluation["issues"]) < len(evaluation["issues"]):
                                        result["tailored_resume"] = fallback_latex
                                        result["evaluation"] = fallback_evaluation
                                        result["summary"] += "\n\nUsed fallback approach to preserve critical sections."
                                        
                                        # Regenerate PDF for fallback version
                                        try:
                                            pdf_base64, pdf_method = compile_latex_to_pdf(fallback_latex)
                                            result["pdf"] = pdf_base64
                                            result["pdf_method"] = pdf_method
                                        except Exception:
                                            pass  # Already logged in compile_latex_to_pdf
                                except Exception as fallback_error:
                                    logger.error(f"Error in fallback approach: {str(fallback_error)}")
                    else:
                        logger.warning("Fix attempt did not return valid LaTeX")
                        
                except Exception as fix_error:
                    logger.error(f"Error attempting fix: {str(fix_error)}")
                
                # Add warning to summary regardless of fix success
                result["format_warning"] = "The resume may have formatting issues. Please check the output carefully."
            else:
                result["format_status"] = "valid"
        
        return jsonify(result), 200
    except Exception as e:
        logger.exception(f"Process endpoint error: {str(e)}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'latex': r"\documentclass{article}\n\usepackage{geometry}\n\geometry{margin=1in}\n\begin{document}\nError processing request.\n\end{document}",
            'pdf': generate_error_pdf(),
            'pdf_method': 'error_pdf',
            'summary': f"An error occurred: {str(e)}"
        }), 500
    
@app.route('/debug/pdf', methods=['POST'])
def debug_pdf_endpoint():
    """Debug endpoint to test PDF extraction capabilities."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
            
        pdf_file = request.files['file']
        pdf_content = pdf_file.read()
        
        # Try direct binary upload
        binary_extraction = extract_text_from_pdf(pdf_content, "direct_binary.pdf")
        
        # Convert to base64 and try base64 method
        base64_data = f"data:application/pdf;base64,{base64.b64encode(pdf_content).decode('utf-8')}"
        base64_extraction = extract_text_from_pdf(base64_data, "base64.pdf")
        
        return jsonify({
            'status': 'success',
            'binary_extraction': {
                'success': binary_extraction is not None,
                'length': len(binary_extraction) if binary_extraction else 0,
                'sample': binary_extraction[:500] if binary_extraction else None
            },
            'base64_extraction': {
                'success': base64_extraction is not None,
                'length': len(base64_extraction) if base64_extraction else 0,
                'sample': base64_extraction[:500] if base64_extraction else None
            }
        }), 200
    except Exception as e:
        logger.exception(f"Debug PDF endpoint error: {str(e)}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/debug/pdf-extraction', methods=['POST'])
def debug_pdf_extraction():
    """Endpoint to test PDF extraction capabilities."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
            
        pdf_file = request.files['file']
        if not pdf_file.filename.lower().endswith('.pdf'):
            return jsonify({'error': 'File must be a PDF'}), 400
            
        # Read the file
        pdf_content = pdf_file.read()
        
        # Perform extraction with multiple methods
        results = {}
        
        # Standard PyPDF2 extraction
        try:
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                temp_path = temp_file.name
                temp_file.write(pdf_content)
            
            standard_text = extract_with_pypdf2(temp_path)
            standard_stats = calculate_text_statistics(standard_text)
            standard_valid, standard_reason = validate_extracted_text(standard_text, "resume")
            
            results['standard_extraction'] = {
                'success': standard_text is not None,
                'text_sample': standard_text[:500] if standard_text else None,
                'stats': standard_stats,
                'valid': standard_valid,
                'validation_reason': standard_reason
            }
            
            # Layout-preserving extraction
            layout_text = extract_with_pypdf2_layout(temp_path)
            layout_stats = calculate_text_statistics(layout_text)
            layout_valid, layout_reason = validate_extracted_text(layout_text, "resume")
            
            results['layout_extraction'] = {
                'success': layout_text is not None,
                'text_sample': layout_text[:500] if layout_text else None,
                'stats': layout_stats,
                'valid': layout_valid,
                'validation_reason': layout_reason
            }
            
            # Combined enhanced extraction
            enhanced_text = extract_text_from_pdf(pdf_content, pdf_file.filename)
            enhanced_stats = calculate_text_statistics(enhanced_text)
            enhanced_valid, enhanced_reason = validate_extracted_text(enhanced_text, "resume")
            
            results['enhanced_extraction'] = {
                'success': enhanced_text is not None,
                'text_sample': enhanced_text[:500] if enhanced_text else None,
                'stats': enhanced_stats,
                'valid': enhanced_valid,
                'validation_reason': enhanced_reason
            }
            
            # Clean up
            os.unlink(temp_path)
            
            return jsonify({
                'status': 'success',
                'filename': pdf_file.filename,
                'filesize': len(pdf_content),
                'results': results
            }), 200
            
        except Exception as e:
            logger.exception(f"Error in PDF extraction debug: {str(e)}")
            return jsonify({
                'status': 'error',
                'error': str(e)
            }), 500
            
    except Exception as e:
        logger.exception(f"PDF extraction debug endpoint error: {str(e)}")
        return jsonify({'status': 'error', 'error': str(e)}), 500

if __name__ == '__main__':
    # Create required directories
    os.makedirs(config.VECTOR_STORE_DIR, exist_ok=True)
    
    logger.info("=== Resume Tailoring App Starting ===")
    logger.info(f"PDF Generators: {', '.join([g.name for g in pdf_generators if g.available])}")
    logger.info(f"Default Model: {config.DEFAULT_MODEL}")
    logger.info(f"Embedding Model: {config.EMBEDDING_MODEL if embeddings else 'Not available'}")
    logger.info(f"GPU Acceleration: {'Enabled' if config.USE_GPU else 'Disabled'}")
    logger.info(f"Vector Store Directory: {config.VECTOR_STORE_DIR}")
    logger.info(f"Retrieval Mode: {config.RETRIEVAL_MODE}")
    logger.info(f"Log Level: {config.LOG_LEVEL}")
    
    # Use waitress or gunicorn in production
    
    app.run(host='0.0.0.0', port=5000)