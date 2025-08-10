"""Web content ingestion using multiple strategies."""

import asyncio
import aiohttp
from typing import List, Dict, Any, Optional
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup

try:
    from crawl4ai import AsyncWebCrawler
    CRAWL4AI_AVAILABLE = True
except ImportError:
    CRAWL4AI_AVAILABLE = False

from .models import Document


class WebIngestion:
    """Web content ingestion with multiple strategies."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.max_concurrent = config.get("max_concurrent", 5)
        self.timeout = config.get("timeout", 30)
        
    async def ingest_url(self, url: str, metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        """Ingest content from a single URL."""
        if CRAWL4AI_AVAILABLE and self.config.get("use_crawl4ai", True):
            return await self._crawl4ai_ingest(url, metadata)
        else:
            return await self._simple_web_ingest(url, metadata)
    
    async def ingest_urls(self, urls: List[str], metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        """Ingest content from multiple URLs."""
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def process_url(url):
            async with semaphore:
                try:
                    return await self.ingest_url(url, metadata)
                except Exception as e:
                    print(f"Error processing {url}: {e}")
                    return []
        
        tasks = [process_url(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        all_documents = []
        for result in results:
            if isinstance(result, list):
                all_documents.extend(result)
        
        return all_documents
    
    async def _crawl4ai_ingest(self, url: str, metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        """Ingest using Crawl4AI for advanced web scraping."""
        async with AsyncWebCrawler(verbose=False) as crawler:
            result = await crawler.arun(
                url=url,
                word_count_threshold=10,
                extraction_strategy="NoExtractionStrategy",
                chunking_strategy="RegexChunking",
                bypass_cache=True
            )
            
            if result.success:
                document = Document(
                    id=f"web_{hash(url) % 10000}",
                    content=result.markdown,
                    metadata={
                        "url": url,
                        "title": result.metadata.get("title", ""),
                        "description": result.metadata.get("description", ""),
                        "keywords": result.metadata.get("keywords", []),
                        "links_count": len(result.links.get("internal", [])) + len(result.links.get("external", [])),
                        "word_count": len(result.markdown.split()),
                        **(metadata or {})
                    },
                    source_tool="crawl4ai",
                    source_id=url,
                    ts_ingested=str(asyncio.get_event_loop().time())
                )
                return [document]
            else:
                raise RuntimeError(f"Failed to crawl {url}: {result.error_message}")
    
    async def _simple_web_ingest(self, url: str, metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        """Simple web scraping fallback."""
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
            async with session.get(url) as response:
                if response.status != 200:
                    raise RuntimeError(f"HTTP {response.status} for {url}")
                
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                # Extract text content
                for script in soup(["script", "style"]):
                    script.decompose()
                
                text = soup.get_text()
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                content = ' '.join(chunk for chunk in chunks if chunk)
                
                # Extract metadata
                title = soup.find('title')
                title_text = title.string if title else ""
                
                meta_desc = soup.find('meta', attrs={'name': 'description'})
                description = meta_desc.get('content', '') if meta_desc else ""
                
                document = Document(
                    id=f"web_simple_{hash(url) % 10000}",
                    content=content,
                    metadata={
                        "url": url,
                        "title": title_text,
                        "description": description,
                        "word_count": len(content.split()),
                        **(metadata or {})
                    },
                    source_tool="simple_web",
                    source_id=url,
                    ts_ingested=str(asyncio.get_event_loop().time())
                )
                
                return [document]
    
    async def ingest_sitemap(self, sitemap_url: str, max_urls: int = 100, 
                           metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        """Ingest URLs from a sitemap."""
        async with aiohttp.ClientSession() as session:
            async with session.get(sitemap_url) as response:
                if response.status != 200:
                    raise RuntimeError(f"Failed to fetch sitemap: {response.status}")
                
                xml_content = await response.text()
                soup = BeautifulSoup(xml_content, 'xml')
                
                # Extract URLs from sitemap
                urls = []
                for loc in soup.find_all('loc'):
                    if len(urls) >= max_urls:
                        break
                    urls.append(loc.text.strip())
                
                # Ingest all URLs
                return await self.ingest_urls(urls, metadata)