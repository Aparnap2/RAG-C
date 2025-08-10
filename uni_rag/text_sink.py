"""
Text sink for chunking, embedding, and indexing documents.
Implements structural chunking, batch embedding, and chunk manifests.
"""
import json
import logging
import hashlib
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

class Chunker:
    """
    Chunks documents into smaller pieces for indexing.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.chunk_size = config.get("chunk_size", 300)  # Target token count
        self.chunk_overlap = config.get("chunk_overlap", 50)  # Token overlap
        self.version = config.get("version", "1.0")
        
    async def chunk_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chunk a document into smaller pieces
        
        Args:
            document: Normalized document
            
        Returns:
            List of chunks
        """
        content = document["content"]
        
        # Simple chunking by paragraphs
        paragraphs = content.split("\n\n")
        
        # Create chunks
        chunks = []
        current_chunk = ""
        current_tokens = 0
        
        for paragraph in paragraphs:
            # Estimate token count (very rough approximation)
            paragraph_tokens = len(paragraph.split())
            
            # If adding this paragraph would exceed chunk size, create a new chunk
            if current_tokens + paragraph_tokens > self.chunk_size and current_chunk:
                # Create chunk
                chunk = self._create_chunk(document, current_chunk)
                chunks.append(chunk)
                
                # Start a new chunk with overlap
                overlap_words = current_chunk.split()[-self.chunk_overlap:]
                current_chunk = " ".join(overlap_words) + "\n\n" + paragraph
                current_tokens = len(overlap_words) + paragraph_tokens
            else:
                # Add paragraph to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
                current_tokens += paragraph_tokens
                
        # Add the last chunk if not empty
        if current_chunk:
            chunk = self._create_chunk(document, current_chunk)
            chunks.append(chunk)
            
        return chunks
        
    def _create_chunk(self, document: Dict[str, Any], text: str) -> Dict[str, Any]:
        """
        Create a chunk from document and text
        
        Args:
            document: Source document
            text: Chunk text
            
        Returns:
            Chunk object
        """
        # Create a deterministic chunk ID
        chunk_id = hashlib.md5(f"{document['id']}:{text}".encode()).hexdigest()
        
        # Estimate token count
        tokens = len(text.split())
        
        # Create chunk
        chunk = {
            "chunk_id": chunk_id,
            "doc_id": document["id"],
            "text": text,
            "tokens": tokens,
            "tenant_id": document["tenant_id"],
            "source_tool": document["source_tool"],
            "source_id": document["source_id"],
            "acl": document["acl"],
            "metadata": document["metadata"],
            "ts_source": document["ts_source"],
            "ts_chunked": datetime.now().isoformat(),
            "chunker_version": self.version
        }
        
        return chunk


class Embedder:
    """
    Generates embeddings for chunks.
    """
    def __init__(self, embedding_client, config: Dict[str, Any]):
        self.embedding_client = embedding_client
        self.config = config
        self.model = config.get("model", "default")
        self.batch_size = config.get("batch_size", 16)
        self.version = config.get("version", "1.0")
        
    async def embed_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate embeddings for chunks
        
        Args:
            chunks: List of chunks
            
        Returns:
            Chunks with embeddings
        """
        # Process in batches
        embedded_chunks = []
        
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i:i+self.batch_size]
            
            # Extract texts
            texts = [chunk["text"] for chunk in batch]
            
            # Generate embeddings
            embeddings = await self.embedding_client.embed_documents(texts, self.model)
            
            # Add embeddings to chunks
            for j, chunk in enumerate(batch):
                chunk["embedding"] = embeddings[j]
                chunk["embedding_model"] = self.model
                chunk["embedding_version"] = self.version
                chunk["ts_embedded"] = datetime.now().isoformat()
                embedded_chunks.append(chunk)
                
        return embedded_chunks


class ChunkManifest:
    """
    Manages chunk manifests for documents.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    async def create_manifest(self, document: Dict[str, Any], chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create a chunk manifest for a document
        
        Args:
            document: Source document
            chunks: Document chunks
            
        Returns:
            Chunk manifest
        """
        # Create manifest
        manifest = {
            "doc_id": document["id"],
            "tenant_id": document["tenant_id"],
            "source_tool": document["source_tool"],
            "source_id": document["source_id"],
            "checksum": document["checksum"],
            "chunk_count": len(chunks),
            "chunk_ids": [chunk["chunk_id"] for chunk in chunks],
            "ts_created": datetime.now().isoformat()
        }
        
        return manifest
        
    async def update_manifest(self, old_manifest: Dict[str, Any], new_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Update a chunk manifest
        
        Args:
            old_manifest: Existing manifest
            new_chunks: New chunks
            
        Returns:
            Updated manifest
        """
        # Create updated manifest
        manifest = old_manifest.copy()
        manifest["chunk_count"] = len(new_chunks)
        manifest["chunk_ids"] = [chunk["chunk_id"] for chunk in new_chunks]
        manifest["ts_updated"] = datetime.now().isoformat()
        
        return manifest


class TextSink:
    """
    Sink for chunking, embedding, and indexing documents.
    """
    def __init__(self, vector_store, text_index, config: Dict[str, Any]):
        self.vector_store = vector_store
        self.text_index = text_index
        self.config = config
        self.chunker = Chunker(config.get("chunker", {}))
        self.embedder = Embedder(config.get("embedding_client"), config.get("embedder", {}))
        self.manifest_manager = ChunkManifest(config)
        
    async def process_document(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a document: chunk, embed, and index
        
        Args:
            document: Normalized document
            
        Returns:
            Processing result
        """
        # Check if document already exists
        existing_manifest = await self._get_manifest(document["id"])
        
        # Chunk the document
        chunks = await self.chunker.chunk_document(document)
        
        # Generate embeddings
        embedded_chunks = await self.embedder.embed_chunks(chunks)
        
        # Create or update manifest
        if existing_manifest:
            # Get old chunks to delete
            old_chunk_ids = set(existing_manifest["chunk_ids"])
            new_chunk_ids = set(chunk["chunk_id"] for chunk in embedded_chunks)
            
            # Chunks to delete
            to_delete = old_chunk_ids - new_chunk_ids
            
            # Delete old chunks
            if to_delete:
                await self._delete_chunks(list(to_delete))
                
            # Update manifest
            manifest = await self.manifest_manager.update_manifest(existing_manifest, embedded_chunks)
        else:
            # Create new manifest
            manifest = await self.manifest_manager.create_manifest(document, embedded_chunks)
            
        # Store manifest
        await self._store_manifest(manifest)
        
        # Index chunks
        await self._index_chunks(embedded_chunks)
        
        return {
            "document_id": document["id"],
            "chunk_count": len(embedded_chunks),
            "manifest_id": document["id"]
        }
        
    async def _get_manifest(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get chunk manifest for a document"""
        # In a real implementation, this would fetch from a database
        # For now, just return None
        return None
        
    async def _store_manifest(self, manifest: Dict[str, Any]) -> None:
        """Store chunk manifest"""
        # In a real implementation, this would store to a database
        logger.info(f"Storing manifest for document {manifest['doc_id']}")
        
    async def _delete_chunks(self, chunk_ids: List[str]) -> None:
        """Delete chunks from indexes"""
        # Delete from vector store
        await self.vector_store.delete(chunk_ids)
        
        # Delete from text index
        await self.text_index.delete(chunk_ids)
        
    async def _index_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """Index chunks in vector store and text index"""
        # Index in vector store
        await self.vector_store.upsert(chunks)
        
        # Index in text index
        await self.text_index.upsert(chunks)
        
    async def process_batch(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process a batch of documents
        
        Args:
            documents: List of normalized documents
            
        Returns:
            List of processing results
        """
        results = []
        
        for document in documents:
            try:
                result = await self.process_document(document)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing document {document['id']} in text sink: {str(e)}")
                # Add error result
                results.append({
                    "document_id": document["id"],
                    "error": str(e)
                })
                
        return results