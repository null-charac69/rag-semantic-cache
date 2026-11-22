"""
RAG Pipeline Service
====================
Orchestrates the entire RAG workflow.

This is the BRAIN that coordinates everything:
1. Check semantic cache
2. If miss → retrieve from vector DB
3. Generate answer with LLM
4. Store in cache
5. Return response

This is where the magic happens!
"""

from typing import Optional, Dict, Any, List
from datetime import datetime
from loguru import logger
import time

from app.models.schemas import (
    QueryRequest,
    QueryResponse,
    CacheStatus,
    SourceChunk,
    DocumentChunk
)
from app.services.cache import get_cache
from app.services.vector_store import get_vector_store
from app.services.llm import get_llm_service
from app.services.embedding import get_embedding_service
from app.services.document_processor import get_document_processor
from app.core.config import settings


class RAGPipeline:
    """
    Complete RAG pipeline orchestrator.
    
    Coordinates all services to answer user queries.
    """
    
    def __init__(self):
        """Initialize all services."""
        logger.info("Initializing RAG pipeline...")
        
        try:
            # Initialize all services
            self.cache = get_cache()
            self.vector_store = get_vector_store()
            self.llm = get_llm_service()
            self.embedding_service = get_embedding_service()
            self.document_processor = get_document_processor()
            
            logger.success("✅ RAG pipeline initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize RAG pipeline: {e}")
            raise
    
    
    def query(self, request: QueryRequest) -> QueryResponse:
        """
        Process user query through complete RAG pipeline.
        
        Flow:
        1. Check semantic cache
        2. If cache hit → return cached response (FAST!)
        3. If cache miss:
           a. Retrieve relevant chunks from vector DB
           b. Generate answer with LLM
           c. Store in cache
        4. Return response
        
        Args:
            request: QueryRequest with query and options
            
        Returns:
            QueryResponse with answer, sources, timing
        """
        start_time = time.time()
        query = request.query
        
        logger.info(f"Processing query: '{query[:50]}...'")
        
        try:
            # ========================================
            # STEP 1: CHECK SEMANTIC CACHE
            # ========================================
            cache_result = None
            if request.use_cache and settings.CACHE_ENABLED:
                logger.info("🔍 Checking semantic cache...")
                cache_result = self.cache.check_cache(query)
            
            # ========================================
            # CACHE HIT PATH (FAST!)
            # ========================================
            if cache_result:
                logger.success("🎯 CACHE HIT - Returning cached response")
                
                processing_time = (time.time() - start_time) * 1000  # ms
                
                return QueryResponse(
                    query=query,
                    answer=cache_result['response'],
                    cache_status=CacheStatus.HIT,
                    sources=cache_result.get('sources') if request.include_sources else None,
                    cache_similarity=cache_result.get('similarity'),
                    processing_time_ms=processing_time,
                    metadata={
                        "cached_query": cache_result.get('cached_query'),
                        "hit_count": cache_result.get('hit_count', 0),
                        "cache_created_at": cache_result.get('created_at').isoformat() if cache_result.get('created_at') else None
                    }
                )
            
            # ========================================
            # CACHE MISS PATH (FULL RAG)
            # ========================================
            logger.info("❌ CACHE MISS - Running full RAG pipeline")
            
            # ========================================
            # STEP 2: RETRIEVE FROM VECTOR DB
            # ========================================
            top_k = request.top_k or settings.TOP_K_RESULTS
            
            logger.info(f"🔎 Searching vector database (top_k={top_k})...")
            source_chunks = self.vector_store.similarity_search(
                query=query,
                top_k=top_k
            )
            
            if not source_chunks:
                logger.warning("⚠️ No relevant chunks found in vector database")
                
                # No context found - return informative response
                processing_time = (time.time() - start_time) * 1000
                
                return QueryResponse(
                    query=query,
                    answer="I don't have any relevant information in my knowledge base to answer this question. Please try uploading relevant documents first.",
                    cache_status=CacheStatus.MISS,
                    sources=None,
                    processing_time_ms=processing_time,
                    metadata={"reason": "no_relevant_chunks"}
                )
            
            logger.info(f"✅ Retrieved {len(source_chunks)} relevant chunks")
            
            # Log retrieval scores for debugging
            for idx, chunk in enumerate(source_chunks, 1):
                logger.debug(
                    f"  Chunk {idx}: score={chunk.score:.3f}, "
                    f"source={chunk.metadata.get('filename', 'unknown')}"
                )
            
            # ========================================
            # STEP 3: GENERATE ANSWER WITH LLM
            # ========================================
            logger.info("🤖 Generating answer with LLM...")
            answer = self.llm.generate_answer(
                query=query,
                context_chunks=source_chunks
            )
            
            logger.success(f"✅ Answer generated ({len(answer)} characters)")
            
            # ========================================
            # STEP 4: STORE IN CACHE
            # ========================================
            if request.use_cache and settings.CACHE_ENABLED:
                logger.info("💾 Storing response in cache...")
                cache_stored = self.cache.store_in_cache(
                    query=query,
                    response=answer,
                    sources=source_chunks if request.include_sources else None
                )
                
                if cache_stored:
                    logger.success("✅ Response cached for future queries")
                else:
                    logger.warning("⚠️ Failed to cache response")
            
            # ========================================
            # STEP 5: RETURN RESPONSE
            # ========================================
            processing_time = (time.time() - start_time) * 1000  # ms
            
            logger.success(
                f"✅ Query processed successfully in {processing_time:.0f}ms"
            )
            
            return QueryResponse(
                query=query,
                answer=answer,
                cache_status=CacheStatus.MISS if request.use_cache else CacheStatus.DISABLED,
                sources=source_chunks if request.include_sources else None,
                processing_time_ms=processing_time,
                metadata={
                    "num_chunks_retrieved": len(source_chunks),
                    "top_chunk_score": source_chunks[0].score if source_chunks else None,
                    "model_used": self.llm.get_model_info()['model']
                }
            )
            
        except Exception as e:
            logger.error(f"❌ Error in RAG pipeline: {e}", exc_info=True)
            
            processing_time = (time.time() - start_time) * 1000
            
            # Return error response instead of crashing
            return QueryResponse(
                query=query,
                answer=f"I encountered an error while processing your query: {str(e)}",
                cache_status=CacheStatus.DISABLED,
                sources=None,
                processing_time_ms=processing_time,
                metadata={"error": str(e)}
            )
    
    
    def ingest_document(
        self,
        file_path: str,
        filename: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Ingest document into the system.
        
        Process:
        1. Process document (extract, chunk)
        2. Generate embeddings
        3. Store in vector database
        
        Args:
            file_path: Path to document file
            filename: Original filename
            metadata: Additional metadata
            
        Returns:
            Dict with ingestion stats
        """
        start_time = time.time()
        
        logger.info(f"📄 Ingesting document: {filename}")
        
        try:
            # ========================================
            # STEP 1: PROCESS DOCUMENT
            # ========================================
            logger.info("📝 Processing document...")
            chunks = self.document_processor.process_document(
                file_path=file_path,
                filename=filename,
                metadata=metadata
            )
            
            if not chunks:
                logger.error("❌ No chunks extracted from document")
                return {
                    "success": False,
                    "error": "Failed to extract content from document"
                }
            
            logger.success(f"✅ Extracted {len(chunks)} chunks")
            
            # ========================================
            # STEP 2: STORE IN VECTOR DATABASE
            # ========================================
            logger.info("💾 Storing chunks in vector database...")
            result = self.vector_store.add_documents(chunks)
            
            # ========================================
            # STEP 3: INVALIDATE RELATED CACHE
            # ========================================
            # When new document added, old cached responses might be outdated
            # Option: Clear entire cache (simple but aggressive)
            # Option: Keep cache (simpler, cached responses still valid for old docs)
            # We'll keep cache for simplicity
            
            processing_time = (time.time() - start_time) * 1000
            
            logger.success(
                f"✅ Document ingested successfully in {processing_time:.0f}ms"
            )
            
            return {
                "success": True,
                "document_id": chunks[0].metadata.get('document_id'),
                "filename": filename,
                "num_chunks": len(chunks),
                "total_characters": sum(len(c.content) for c in chunks),
                "document_type": chunks[0].metadata.get('document_type'),
                "processing_time_ms": processing_time,
                "vector_store_count": result.get('total_items', 0)
            }
            
        except Exception as e:
            logger.error(f"❌ Error ingesting document: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }
    
    
    def get_system_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the RAG system.
        
        Useful for monitoring and debugging.
        
        Returns:
            Dict with system statistics
        """
        try:
            return {
                "vector_store": {
                    "total_chunks": self.vector_store.get_document_count(),
                    "collection_name": settings.COLLECTION_NAME
                },
                "cache": self.cache.get_cache_stats(),
                "llm": self.llm.get_model_info(),
                "embedding": {
                    "model": self.embedding_service.get_model_name(),
                    "dimension": self.embedding_service.get_dimension()
                },
                "config": {
                    "chunk_size": settings.CHUNK_SIZE,
                    "chunk_overlap": settings.CHUNK_OVERLAP,
                    "top_k": settings.TOP_K_RESULTS,
                    "cache_threshold": settings.CACHE_SIMILARITY_THRESHOLD
                }
            }
        except Exception as e:
            logger.error(f"Error getting system stats: {e}")
            return {"error": str(e)}
    
    
    def health_check(self) -> Dict[str, str]:
        """
        Perform health check on all services.
        
        Returns:
            Dict with service statuses
        """
        health = {
            "status": "healthy",
            "services": {}
        }
        
        try:
            # Check vector store
            try:
                count = self.vector_store.get_document_count()
                health["services"]["vector_store"] = f"healthy ({count} chunks)"
            except Exception as e:
                health["services"]["vector_store"] = f"unhealthy: {str(e)}"
                health["status"] = "degraded"
            
            # Check cache
            try:
                cache_stats = self.cache.get_cache_stats()
                if cache_stats.get('enabled'):
                    health["services"]["cache"] = "healthy"
                else:
                    health["services"]["cache"] = "disabled"
            except Exception as e:
                health["services"]["cache"] = f"unhealthy: {str(e)}"
            
            # Check LLM
            try:
                model_info = self.llm.get_model_info()
                health["services"]["llm"] = f"healthy (model: {model_info['model']})"
            except Exception as e:
                health["services"]["llm"] = f"unhealthy: {str(e)}"
                health["status"] = "degraded"
            
            # Check embedding service
            try:
                dimension = self.embedding_service.get_dimension()
                health["services"]["embedding"] = f"healthy (dim: {dimension})"
            except Exception as e:
                health["services"]["embedding"] = f"unhealthy: {str(e)}"
                health["status"] = "degraded"
            
        except Exception as e:
            health["status"] = "unhealthy"
            health["error"] = str(e)
        
        return health
    
    
    def clear_cache(self) -> Dict[str, Any]:
        """
        Clear the semantic cache.
        
        Useful after:
        - Updating documents
        - Changing LLM model
        - Testing
        
        Returns:
            Dict with clear stats
        """
        try:
            logger.warning("⚠️ Clearing semantic cache...")
            count = self.cache.invalidate_cache()
            logger.success(f"✅ Cleared {count} cache entries")
            
            return {
                "success": True,
                "entries_cleared": count
            }
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    
    def delete_document(self, document_id: str) -> Dict[str, Any]:
        """
        Delete document from vector store.
        
        Args:
            document_id: Document identifier
            
        Returns:
            Dict with deletion stats
        """
        try:
            logger.info(f"🗑️ Deleting document: {document_id}")
            result = self.vector_store.delete_document(document_id)
            
            # Optionally clear cache when document deleted
            if result.get('deleted_count', 0) > 0:
                logger.info("Clearing cache due to document deletion")
                self.cache.invalidate_cache()
            
            logger.success(f"✅ Deleted {result.get('deleted_count', 0)} chunks")
            
            return {
                "success": True,
                **result
            }
        except Exception as e:
            logger.error(f"Error deleting document: {e}")
            return {
                "success": False,
                "error": str(e)
            }


# ============================================================
# SINGLETON INSTANCE
# ============================================================

_pipeline_instance = None

def get_rag_pipeline() -> RAGPipeline:
    """
    Get or create singleton RAG pipeline.
    
    This is THE main entry point for all RAG operations.
    
    Usage:
        from app.services.rag_pipeline import get_rag_pipeline
        
        pipeline = get_rag_pipeline()
        
        # Query
        response = pipeline.query(QueryRequest(query="What is ML?"))
        
        # Ingest document
        result = pipeline.ingest_document("paper.pdf", "paper.pdf")
        
        # Health check
        health = pipeline.health_check()
    """
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = RAGPipeline()
    return _pipeline_instance


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def quick_query(query: str, use_cache: bool = True) -> str:
    """
    Quick query helper for simple use cases.
    
    Returns just the answer string (not full response object).
    
    Args:
        query: User's question
        use_cache: Whether to use cache
        
    Returns:
        Answer string
    """
    pipeline = get_rag_pipeline()
    request = QueryRequest(query=query, use_cache=use_cache)
    response = pipeline.query(request)
    return response.answer


def quick_ingest(file_path: str) -> bool:
    """
    Quick document ingestion helper.
    
    Args:
        file_path: Path to document
        
    Returns:
        True if successful
    """
    from pathlib import Path
    
    pipeline = get_rag_pipeline()
    filename = Path(file_path).name
    
    result = pipeline.ingest_document(file_path, filename)
    return result.get('success', False)
