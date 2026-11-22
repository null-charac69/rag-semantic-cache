"""
Vector Store Service
====================
Manages vector database operations using ChromaDB.

What is a vector database?
- Stores embeddings (vectors) with metadata
- Enables similarity search (find nearest neighbors)
- Think: "Google for semantics" instead of keywords

ChromaDB advantages:
- Lightweight (perfect for personal projects)
- Persistent storage (survives restarts)
- Built-in similarity search
- No separate server needed (embedded mode)
"""

import chromadb
from chromadb.config import Settings as ChromaSettings
from typing import List, Dict, Any, Optional
from loguru import logger
import numpy as np

from app.core.config import settings
from app.models.schemas import DocumentChunk, SourceChunk
from app.services.embedding import get_embedding_service


class VectorStore:
    """
    ChromaDB vector store wrapper.
    
    Handles:
    - Storing document chunks with embeddings
    - Similarity search (retrieve relevant chunks)
    - Collection management
    """
    
    def __init__(self):
        """
        Initialize ChromaDB client and collection.
        
        Two modes:
        1. Persistent (saves to disk) ← We use this
        2. In-memory (lost on restart)
        """
        logger.info("Initializing ChromaDB vector store...")
        
        try:
            # Create persistent client
            self.client = chromadb.PersistentClient(
                path=settings.CHROMA_PERSIST_DIR,
                settings=ChromaSettings(
                    anonymized_telemetry=False,  # Disable analytics
                    allow_reset=True  # Allow clearing database
                )
            )
            
            # Get embedding service
            self.embedding_service = get_embedding_service()
            
            # Get or create collection
            # Collection = like a "table" in SQL
            self.collection = self.client.get_or_create_collection(
                name=settings.COLLECTION_NAME,
                metadata={
                    "description": "RAG document chunks",
                    "embedding_model": settings.EMBEDDING_MODEL,
                    "dimension": self.embedding_service.get_dimension()
                }
            )
            
            logger.success(
                f"✅ Vector store initialized: "
                f"collection='{settings.COLLECTION_NAME}', "
                f"items={self.collection.count()}"
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize vector store: {e}")
            raise
    
    
    def add_documents(self, chunks: List[DocumentChunk]) -> Dict[str, Any]:
        """
        Add document chunks to vector store.
        
        Process:
        1. Extract text from chunks
        2. Generate embeddings (batch for speed!)
        3. Store in ChromaDB with metadata
        
        Args:
            chunks: List of DocumentChunk objects
            
        Returns:
            Dict with stats (count, ids, etc.)
        """
        if not chunks:
            logger.warning("No chunks to add")
            return {"count": 0}
        
        logger.info(f"Adding {len(chunks)} chunks to vector store...")
        
        try:
            # Extract data from chunks
            ids = [chunk.chunk_id for chunk in chunks]
            documents = [chunk.content for chunk in chunks]
            metadatas = [chunk.metadata for chunk in chunks]
            
            # Generate embeddings in batch (FAST!)
            logger.info("Generating embeddings...")
            embeddings = self.embedding_service.embed_batch(documents)
            
            # Add to ChromaDB
            # ChromaDB expects: ids, embeddings, documents, metadatas
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            
            logger.success(f"✅ Added {len(chunks)} chunks to vector store")
            
            return {
                "count": len(chunks),
                "ids": ids,
                "total_items": self.collection.count()
            }
            
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise
    
    
    def similarity_search(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[SourceChunk]:
        """
        Search for similar chunks using semantic similarity.
        
        This is the CORE of RAG retrieval!
        
        Process:
        1. Embed the query
        2. Find most similar vectors in database
        3. Return top-k results with scores
        
        Args:
            query: User's question
            top_k: How many results to return
            filter_metadata: Optional filters (e.g., {"document_type": "pdf"})
            
        Returns:
            List of SourceChunk objects with content + scores
            
        Example:
            Query: "What is machine learning?"
            Returns:
            [
                SourceChunk(
                    content="Machine learning is a subset of AI...",
                    score=0.92,
                    metadata={"filename": "ml_paper.pdf", "page": 1}
                ),
                SourceChunk(
                    content="ML algorithms learn from data...",
                    score=0.87,
                    metadata={"filename": "ai_textbook.docx", "page": 45}
                ),
                ...
            ]
        """
        top_k = top_k or settings.TOP_K_RESULTS
        
        logger.info(f"Searching for: '{query}' (top_k={top_k})")
        
        try:
            # Step 1: Embed the query
            query_embedding = self.embedding_service.embed_text(query)
            
            # Step 2: Search ChromaDB
            # ChromaDB finds nearest neighbors using cosine similarity
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=filter_metadata,  # Optional filtering
                include=["documents", "metadatas", "distances"]
            )
            
            # Step 3: Parse results
            # ChromaDB returns nested lists: [[doc1, doc2, ...]]
            # We flatten and convert to SourceChunk objects
            
            if not results['ids'][0]:  # No results found
                logger.warning("No similar documents found")
                return []
            
            source_chunks = []
            
            # Iterate through results
            for idx in range(len(results['ids'][0])):
                # ChromaDB returns "distance" (lower = more similar)
                # Convert to similarity score (0-1, higher = more similar)
                distance = results['distances'][0][idx]
                similarity_score = self._distance_to_similarity(distance)
                
                chunk = SourceChunk(
                    content=results['documents'][0][idx],
                    score=similarity_score,
                    metadata=results['metadatas'][0][idx],
                    document_id=results['ids'][0][idx]
                )
                
                source_chunks.append(chunk)
            
            logger.success(
                f"✅ Found {len(source_chunks)} relevant chunks "
                f"(scores: {[f'{c.score:.2f}' for c in source_chunks]})"
            )
            
            return source_chunks
            
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            raise
    
    
    def _distance_to_similarity(self, distance: float) -> float:
        """
        Convert ChromaDB distance to similarity score.
        
        ChromaDB uses L2 (Euclidean) distance by default:
        - Distance 0 = identical vectors
        - Larger distance = less similar
        
        We convert to 0-1 similarity score:
        - 1.0 = identical
        - 0.0 = completely different
        
        Formula: similarity = 1 / (1 + distance)
        
        Examples:
        - distance=0 → similarity=1.0 (perfect match!)
        - distance=1 → similarity=0.5 (somewhat similar)
        - distance=10 → similarity=0.09 (very different)
        """
        # Ensure non-negative
        distance = max(0, distance)
        
        # Convert to similarity
        similarity = 1.0 / (1.0 + distance)
        
        return float(similarity)
    
    
    def delete_document(self, document_id: str) -> Dict[str, Any]:
        """
        Delete all chunks from a specific document.
        
        Useful for:
        - Removing outdated documents
        - Cleaning up test data
        - Implementing document updates (delete + re-add)
        
        Args:
            document_id: Document identifier
            
        Returns:
            Dict with deletion stats
        """
        try:
            logger.info(f"Deleting document: {document_id}")
            
            # Find all chunks with this document_id
            results = self.collection.get(
                where={"document_id": document_id},
                include=[]
            )
            
            if not results['ids']:
                logger.warning(f"No chunks found for document: {document_id}")
                return {"deleted_count": 0}
            
            # Delete all chunks
            self.collection.delete(ids=results['ids'])
            
            deleted_count = len(results['ids'])
            logger.success(f"✅ Deleted {deleted_count} chunks")
            
            return {
                "deleted_count": deleted_count,
                "document_id": document_id
            }
            
        except Exception as e:
            logger.error(f"Error deleting document: {e}")
            raise
    
    
    def get_document_count(self) -> int:
        """
        Get total number of chunks in vector store.
        
        Useful for monitoring and health checks.
        """
        try:
            return self.collection.count()
        except Exception as e:
            logger.error(f"Error getting count: {e}")
            return 0
    
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the collection.
        
        Returns metadata, stats, etc.
        """
        try:
            return {
                "name": self.collection.name,
                "count": self.collection.count(),
                "metadata": self.collection.metadata
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {}
    
    
    def clear_collection(self) -> bool:
        """
        Delete ALL documents from collection.
        
        ⚠️ DANGEROUS! Use only for testing or complete resets.
        """
        try:
            logger.warning("⚠️ Clearing entire collection...")
            
            # Get all IDs
            all_items = self.collection.get(include=[])
            
            if all_items['ids']:
                self.collection.delete(ids=all_items['ids'])
                logger.success(f"✅ Cleared {len(all_items['ids'])} items")
            
            return True
            
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            return False
    
    
    def update_document(
        self,
        document_id: str,
        new_chunks: List[DocumentChunk]
    ) -> Dict[str, Any]:
        """
        Update a document (delete old, add new).
        
        Useful for document versioning.
        
        Args:
            document_id: Document to update
            new_chunks: New chunks to replace with
            
        Returns:
            Dict with update stats
        """
        try:
            logger.info(f"Updating document: {document_id}")
            
            # Delete old version
            delete_result = self.delete_document(document_id)
            
            # Add new version
            add_result = self.add_documents(new_chunks)
            
            return {
                "deleted_count": delete_result.get("deleted_count", 0),
                "added_count": add_result.get("count", 0),
                "document_id": document_id
            }
            
        except Exception as e:
            logger.error(f"Error updating document: {e}")
            raise


# ============================================================
# SINGLETON INSTANCE
# ============================================================

_vector_store_instance = None

def get_vector_store() -> VectorStore:
    """
    Get or create singleton vector store.
    
    Why singleton?
    - ChromaDB client is expensive to create
    - Maintains connection to database
    - Single source of truth
    
    Usage:
        from app.services.vector_store import get_vector_store
        
        store = get_vector_store()
        results = store.similarity_search("What is AI?")
    """
    global _vector_store_instance
    if _vector_store_instance is None:
        _vector_store_instance = VectorStore()
    return _vector_store_instance


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def search_with_reranking(
    query: str,
    top_k: int = 10,
    final_k: int = 3
) -> List[SourceChunk]:
    """
    Advanced search with re-ranking.
    
    Strategy:
    1. Retrieve top_k results (cast wide net)
    2. Re-rank using more sophisticated scoring
    3. Return final_k best results
    
    This improves retrieval quality!
    
    Args:
        query: Search query
        top_k: Initial retrieval count
        final_k: Final results to return
        
    Returns:
        Re-ranked source chunks
    """
    store = get_vector_store()
    embedding_service = get_embedding_service()
    
    # Step 1: Get initial results
    initial_results = store.similarity_search(query, top_k=top_k)
    
    if not initial_results:
        return []
    
    # Step 2: Re-rank
    # Simple re-ranking: recalculate similarity with fresh embedding
    query_embedding = embedding_service.embed_text(query)
    
    for chunk in initial_results:
        # Get chunk embedding
        chunk_embedding = embedding_service.embed_text(chunk.content)
        
        # Recalculate similarity
        similarity = embedding_service.compute_similarity(
            query_embedding,
            chunk_embedding
        )
        
        # Update score
        chunk.score = similarity
    
    # Step 3: Sort by new scores and return top final_k
    initial_results.sort(key=lambda x: x.score, reverse=True)
    
    return initial_results[:final_k]
