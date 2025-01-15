import os
import sys
import asyncio
import requests
from xml.etree import ElementTree
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urlparse
from dotenv import load_dotenv

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

# Load environment variables
load_dotenv()

# Initialize embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Initialize Chroma vector store
VECTOR_STORE_DIR = "./vectorstore"
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
chroma_db = Chroma(persist_directory=VECTOR_STORE_DIR, embedding_function=embedding_model)

@dataclass
class ProcessedChunk:
    url: str
    chunk_number: int
    title: str
    summary: str
    content: str
    metadata: Dict[str, Any]
    embedding: List[float]

def chunk_text(text: str, chunk_size: int = 1000) -> List[str]:
    """Split text into smaller chunks while preserving structure."""
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
        if end >= text_length:
            chunks.append(text[start:].strip())
            break

        # Prefer splitting at sentence endings
        chunk = text[start:end]
        last_period = chunk.rfind('. ')
        if last_period > chunk_size * 0.3:
            end = start + last_period + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = max(start + 1, end)
    
    return chunks

async def get_embedding(text: str) -> List[float]:
    """Generate embedding for a text chunk."""
    return embedding_model.embed_query(text)

async def process_chunk(chunk: str, chunk_number: int, url: str) -> ProcessedChunk:
    """Process text chunk to generate metadata and embedding."""
    embedding = await get_embedding(chunk)
    metadata = {
        "source": "langchain_docs",
        "chunk_size": len(chunk),
        "crawled_at": datetime.now(timezone.utc).isoformat(),
        "url_path": urlparse(url).path
    }
    return ProcessedChunk(
        url=url,
        chunk_number=chunk_number,
        title=f"Chunk {chunk_number} from {url}",
        summary=chunk[:200],
        content=chunk,
        metadata=metadata,
        embedding=embedding
    )

async def insert_chunk_to_chroma(chunk: ProcessedChunk):
    """Insert the processed chunk into ChromaDB."""
    try:
        chroma_db.add_texts(
            texts=[chunk.content],
            metadatas=[chunk.metadata],
            ids=[f"{chunk.url}_chunk_{chunk.chunk_number}"]
        )
        print(f"Inserted chunk {chunk.chunk_number} for {chunk.url}")
    except Exception as e:
        print(f"Error inserting chunk into Chroma: {e}")

async def process_and_store_document(url: str, markdown: str):
    """Process and store all chunks of a document."""
    chunks = chunk_text(markdown)
    tasks = [process_chunk(chunk, i, url) for i, chunk in enumerate(chunks)]
    processed_chunks = await asyncio.gather(*tasks)
    insert_tasks = [insert_chunk_to_chroma(chunk) for chunk in processed_chunks]
    await asyncio.gather(*insert_tasks)

async def crawl_parallel(urls: List[str], max_concurrent: int = 100):
    """Crawl URLs in parallel and store their content."""
    browser_config = BrowserConfig(
        headless=True,
        verbose=False,
        extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"],
    )
    crawl_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)

    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.start()

    try:
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_url(url: str):
            async with semaphore:
                result = await crawler.arun(
                    url=url,
                    config=crawl_config,
                    session_id="session1"
                )
                if result.success:
                    print(f"Successfully crawled: {url}")
                    await process_and_store_document(url, result.markdown_v2.raw_markdown)
                else:
                    print(f"Failed to crawl {url}: {result.error_message}")

        await asyncio.gather(*[process_url(url) for url in urls])
    finally:
        await crawler.close()


##get sitemaps.xml for the documentation you want to build


def get_langchain_docs_urls() -> List[str]:
    """Fetch URLs from the LangChain documentation sitemap."""
    sitemap_url = "https://docs.crewai.com/sitemap.xml"
    try:
        response = requests.get(sitemap_url)
        response.raise_for_status()
        root = ElementTree.fromstring(response.content)
        namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        urls = [loc.text for loc in root.findall('.//ns:loc', namespace)]
        return urls
    except Exception as e:
        print(f"Error fetching sitemap: {e}")
        return []

async def main():
    """Main function to orchestrate crawling and storing."""
    urls = get_langchain_docs_urls()
    if not urls:
        print("No URLs found to crawl.")
        return
    print(f"Found {len(urls)} URLs to crawl.")
    await crawl_parallel(urls)

if __name__ == "__main__":
    asyncio.run(main())
