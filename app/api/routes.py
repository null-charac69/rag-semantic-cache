"""
API Routes
==========
FastAPI endpoints for the RAG system.

Endpoints:
- POST /query - Ask questions
- POST /upload - Upload documents
- GET /health - Health check
- GET /stats - System statistics
- DELETE /cache - Clear cache
- DELETE /document/{doc_id} - Delete document
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from typing import Optional
import os
import shutil
from pathlib import Path
from loguru import logger

from app.models.schemas import (
    QueryRequest,
    QueryResponse,
    DocumentUploadResponse,
    HealthCheck,
    ErrorResponse
)
from app.services.rag_pipeline import get_rag_pipeline
from app.core.config import settings


# Create router
router = APIRouter()


# ============================================================
# QUERY ENDPOINT
# ============================================================

@router.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """
    Ask a question using RAG.
    
    Process:
    1. Check semantic cache
    2. If miss, retrieve relevant chunks
    3. Generate answer with LLM
    4. Cache response
    
    Example request:
```json
    {
        "query": "What is machine learning?",
        "use_cache": true,
        "top_k": 3,
        "include_sources": true
    }
```
    
    Example response:
```json
    {
        "query": "What is machine learning?",
        "answer": "Machine learning is a subset of AI...",
        "cache_status": "miss",
        "sources": [...],
        "processing_time_ms": 2150.5,
        "timestamp": "2024-02-08T10:30:00"
    }
```
    """
    try:
        logger.info(f"API: Received query: '{request.query[:50]}...'")
        
        # Get pipeline and process query
        pipeline = get_rag_pipeline()
        response = pipeline.query(request)
        
        logger.info(
            f"API: Query processed - "
            f"cache_status={response.cache_status.value}, "
            f"time={response.processing_time_ms:.0f}ms"
        )
        
        return response
        
    except Exception as e:
        logger.error(f"API: Error processing query: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )


# ============================================================
# UPLOAD ENDPOINT
# ============================================================

@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    metadata: Optional[str] = Form(None)
):
    """
    Upload and ingest a document.
    
    Supports:
    - PDF, DOCX, PPTX, XLSX
    - TXT, MD, JSON, CSV
    - Code files (.py, .js, etc.)
    - HTML
    
    Example usage (curl):
```bash
    curl -X POST "http://localhost:8000/upload" \
         -F "file=@document.pdf" \
         -F "metadata={\"author\":\"John\"}"
```
    
    Example response:
```json
    {
        "document_id": "a3f5b2c8",
        "filename": "document.pdf",
        "document_type": "pdf",
        "num_chunks": 25,
        "total_characters": 25000,
        "processing_time_ms": 3500,
        "status": "success"
    }
```
    """
    try:
        logger.info(f"API: Received file upload: {file.filename}")
        
        # Validate file size
        file_size = 0
        chunk_size = 1024 * 1024  # 1MB chunks
        
        # Read file to check size
        temp_content = await file.read()
        file_size = len(temp_content)
        
        max_size = settings.MAX_FILE_SIZE_MB * 1024 * 1024
        if file_size > max_size:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Max size: {settings.MAX_FILE_SIZE_MB}MB"
            )
        
        logger.info(f"File size: {file_size / 1024 / 1024:.2f}MB")
        
        # Save file temporarily
        temp_dir = Path("./data/documents")
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        temp_file_path = temp_dir / file.filename
        
        # Write content
        with open(temp_file_path, "wb") as f:
            f.write(temp_content)
        
        logger.info(f"File saved to: {temp_file_path}")
        
        # Parse metadata if provided
        import json
        metadata_dict = None
        if metadata:
            try:
                metadata_dict = json.loads(metadata)
            except json.JSONDecodeError:
                logger.warning("Invalid metadata JSON, ignoring")
        
        # Ingest document
        pipeline = get_rag_pipeline()
        result = pipeline.ingest_document(
            file_path=str(temp_file_path),
            filename=file.filename,
            metadata=metadata_dict
        )
        
        if not result.get('success'):
            raise HTTPException(
                status_code=500,
                detail=result.get('error', 'Failed to process document')
            )
        
        logger.success(f"API: Document ingested successfully: {file.filename}")
        
        # Return response
        return DocumentUploadResponse(
            document_id=result['document_id'],
            filename=result['filename'],
            document_type=result['document_type'],
            num_chunks=result['num_chunks'],
            total_characters=result['total_characters'],
            processing_time_ms=result['processing_time_ms'],
            status="success",
            message=f"Document processed and indexed successfully ({result['num_chunks']} chunks created)"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"API: Error uploading document: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing document: {str(e)}"
        )


# ============================================================
# HEALTH CHECK ENDPOINT
# ============================================================

@router.get("/health", response_model=HealthCheck)
async def health_check():
    """
    Check system health.
    
    Returns status of all services:
    - Vector database
    - Cache
    - LLM service
    - Embedding service
    
    Example response:
```json
    {
        "status": "healthy",
        "timestamp": "2024-02-08T10:30:00",
        "version": "1.0.0",
        "services": {
            "vector_store": "healthy (125 chunks)",
            "cache": "healthy",
            "llm": "healthy (model: llama-3.1-70b-versatile)",
            "embedding": "healthy (dim: 384)"
        }
    }
```
    """
    try:
        pipeline = get_rag_pipeline()
        health = pipeline.health_check()
        
        return HealthCheck(
            status=health['status'],
            version="1.0.0",
            services=health.get('services', {})
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthCheck(
            status="unhealthy",
            version="1.0.0",
            services={"error": str(e)}
        )


# ============================================================
# STATISTICS ENDPOINT
# ============================================================

@router.get("/stats")
async def get_stats():
    """
    Get system statistics.
    
    Returns:
    - Vector store stats (chunk count)
    - Cache stats (entry count, hit rate)
    - LLM model info
    - Embedding model info
    - Configuration
    
    Example response:
```json
    {
        "vector_store": {
            "total_chunks": 125,
            "collection_name": "rag_documents"
        },
        "cache": {
            "enabled": true,
            "total_entries": 23,
            "threshold": 0.95
        },
        "llm": {
            "model": "llama-3.1-70b-versatile",
            "provider": "Groq"
        },
        "config": {
            "chunk_size": 1000,
            "top_k": 3
        }
    }
```
    """
    try:
        pipeline = get_rag_pipeline()
        stats = pipeline.get_system_stats()
        return JSONResponse(content=stats)
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving stats: {str(e)}"
        )


# ============================================================
# CACHE MANAGEMENT ENDPOINTS
# ============================================================

@router.delete("/cache")
async def clear_cache():
    """
    Clear semantic cache.
    
    Use when:
    - Documents updated
    - Model changed
    - Testing
    
    Example response:
```json
    {
        "success": true,
        "entries_cleared": 23,
        "message": "Cache cleared successfully"
    }
```
    """
    try:
        logger.warning("API: Cache clear requested")
        
        pipeline = get_rag_pipeline()
        result = pipeline.clear_cache()
        
        return {
            "success": result['success'],
            "entries_cleared": result.get('entries_cleared', 0),
            "message": "Cache cleared successfully" if result['success'] else "Failed to clear cache"
        }
        
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error clearing cache: {str(e)}"
        )


# ============================================================
# DOCUMENT MANAGEMENT ENDPOINTS
# ============================================================

@router.delete("/document/{document_id}")
async def delete_document(document_id: str):
    """
    Delete document from vector store.
    
    Args:
        document_id: Document identifier
    
    Example response:
```json
    {
        "success": true,
        "deleted_count": 25,
        "document_id": "a3f5b2c8",
        "message": "Document deleted successfully"
    }
```
    """
    try:
        logger.info(f"API: Delete document requested: {document_id}")
        
        pipeline = get_rag_pipeline()
        result = pipeline.delete_document(document_id)
        
        if not result.get('success'):
            raise HTTPException(
                status_code=404,
                detail="Document not found or error deleting"
            )
        
        return {
            **result,
            "message": f"Document deleted successfully ({result.get('deleted_count', 0)} chunks removed)"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting document: {str(e)}"
        )


# ============================================================
# LIST DOCUMENTS ENDPOINT (Optional)
# ============================================================

@router.get("/documents")
async def list_documents():
    """
    List all documents in the system.
    
    Returns summary of indexed documents.
    
    Note: This is a basic implementation.
    For production, consider adding pagination.
    """
    try:
        pipeline = get_rag_pipeline()
        vector_store = pipeline.vector_store
        
        # Get all items from collection
        # Note: This can be slow with many documents
        collection_info = vector_store.get_collection_info()
        
        return {
            "total_chunks": collection_info.get('count', 0),
            "collection_name": collection_info.get('name'),
            "metadata": collection_info.get('metadata')
        }
        
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error listing documents: {str(e)}"
        )


# ============================================================
# TEST ENDPOINT (Development only)
# ============================================================

@router.get("/test")
async def test_endpoint():
    """
    Simple test endpoint.
    
    Returns:
        Test message
    """
    return {
        "message": "RAG API is running!",
        "version": "1.0.0",
        "endpoints": {
            "query": "POST /query",
            "upload": "POST /upload",
            "health": "GET /health",
            "stats": "GET /stats"
        }
    }
