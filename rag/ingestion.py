from rag.interfaces import IngestionBase
from docling.document_converter import DocumentConverter
from pypdf import PdfReader
from crawl4ai import AsyncWebCrawler
from rag.models import Document
from uuid import uuid4
from typing import List

class PDFIngestion(IngestionBase):
    async def ingest(self, file_path: str, **kwargs) -> List[Document]:
        try:
            converter = DocumentConverter()
            doc_result = converter.convert(file_path)
            content = doc_result.document.export_to_markdown()
            return [Document(id=str(uuid4()), content=content, metadata={"source": file_path})]
        except Exception:
            reader = PdfReader(file_path)
            content = "".join(page.extract_text() for page in reader.pages)
            return [Document(id=str(uuid4()), content=content, metadata={"source": file_path})]

class WebIngestion(IngestionBase):
    async def ingest(self, url: str, **kwargs) -> List[Document]:
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url, cache_mode="BYPASS")
            return [Document(id=str(uuid4()), content=result.markdown, metadata={"source": url})]

# Registry for ingestion types
INGESTION_REGISTRY = {
    "pdf": PDFIngestion(),
    "web": WebIngestion(),
}

def get_ingestion_instance(source_type: str):
    if source_type in INGESTION_REGISTRY:
        return INGESTION_REGISTRY[source_type]
    raise ValueError(f"Unknown ingestion type: {source_type}") 