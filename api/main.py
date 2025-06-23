from fastapi import FastAPI, UploadFile, File
from rag.pipeline import RAGPipeline
from rag.models import RAGQuery, RAGResponse
from rag.config import RAGConfig
import os
import shutil
import json

app = FastAPI()

# Load config from file or environment variable
CONFIG_PATH = os.getenv("RAG_CONFIG_PATH", "rag_config.json")
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH) as f:
        config_dict = json.load(f)
    config = RAGConfig(**config_dict)
else:
    config = RAGConfig(
        ingestion={"pdf": {}, "web": {}},
        vector_store={"provider": "astradb"},
        knowledge_graph=None,
        memory=None,
        llm={"provider": "google_genai", "model": "gemini-1.5-pro"},
        prompt_template=None
    )
pipeline = RAGPipeline(config)

@app.post("/api/rag/query", response_model=RAGResponse)
async def query_rag(query: RAGQuery):
    return await pipeline.query(query)

@app.post("/api/rag/ingest")
async def ingest_file(file: UploadFile = File(...)):
    file_path = f"tmp/{file.filename}"
    os.makedirs("tmp", exist_ok=True)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    documents = await pipeline.ingest(source_type="pdf", file_path=file_path)
    os.remove(file_path)
    return {"documents": [doc.model_dump() for doc in documents]} 