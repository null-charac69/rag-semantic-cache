"""
Pydantic Models (Schemas)
=========================
These are the "contracts" for our API - they define what data looks like.

Why Pydantic models?
1. Automatic validation (wrong type = error before code runs)
2. Auto-generated API documentation (FastAPI reads these)
3. Type hints = better IDE autocomplete
4. Serialization/deserialization (Python dict ↔ JSON)
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


# ============================================================
# ENUMS - Predefined choices
# ============================================================

class DocumentType(str, Enum):
    """
    Supported document types.
    Using Enum ensures only valid types are accepted.
    """
    PDF = "pdf"
    DOCX = "docx"
    PPTX = "pptx"
    TXT = "txt"
    MD = "md"
    CSV = "csv"
    JSON = "json"
    HTML = "html"
    XLSX = "xlsx"
    CODE = "code"  # .py, .js, .java, etc.
    UNKNOWN = "unknown"


class CacheStatus(str, Enum):
    """Cache hit status for responses"""
    HIT = "hit"      # Found in cache
    MISS = "miss"    # Not in cache, ran RAG pipeline
    DISABLED = "disabled"  # Caching turned off


# ============================================================
# REQUEST MODELS - What user sends to API
# ============================================================

class QueryRequest(BaseModel):
    """
    User query request schema.
    
    Example JSON:
    {
        "query": "What is retrieval augmented generation?",
        "use_cache": true,
        "top_k": 3
    }
    """
    query: str = Field(
        ...,  # Required field
        min_length=1,
        max_length=2000,
        description="User's question or search query"
    )
    # Validation: query must be 1-2000 characters
    
    use_cache: bool = Field(
        default=True,
        description="Whether to check semantic cache before RAG"
    )
    
    top_k: Optional[int] = Field(
        default=None,  # Will use settings.TOP_K_RESULTS if None
        ge=1,  # Greater than or equal to 1
        le=10,  # Less than or equal to 10
        description="Number of relevant chunks to retrieve"
    )
    
    include_sources: bool = Field(
        default=True,
        description="Include source documents in response"
    )
    
    @validator('query')
    def query_not_empty(cls, v):
        """Custom validator - ensures query isn't just whitespace"""
        if not v.strip():
            raise ValueError("Query cannot be empty or just whitespace")
        return v.strip()


class DocumentUploadMetadata(BaseModel):
    """
    Metadata accompanying document uploads.
    
    Used when uploading files via API.
    """
    filename: str = Field(..., description="Original filename")
    
    document_type: Optional[DocumentType] = Field(
        default=None,
        description="Document type (auto-detected if not provided)"
    )
    
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional metadata (author, date, tags, etc.)"
    )
    # default_factory=dict means "create empty dict if not provided"
    
    chunk_size: Optional[int] = Field(
        default=None,
        description="Override default chunk size for this document"
    )


# ============================================================
# RESPONSE MODELS - What API returns to user
# ============================================================

class SourceChunk(BaseModel):
    """
    A retrieved chunk from vector database.
    
    This is ONE piece of relevant context found.
    """
    content: str = Field(..., description="The actual text content")
    
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Similarity score (0-1, higher = more relevant)"
    )
    # Cosine similarity between query and chunk
    
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Source metadata (filename, page, etc.)"
    )
    
    document_id: Optional[str] = Field(
        default=None,
        description="Unique ID of source document"
    )


class QueryResponse(BaseModel):
    """
    Complete response to user query.
    
    This is what gets returned from /query endpoint.
    """
    query: str = Field(..., description="Original user query")
    
    answer: str = Field(..., description="Generated answer from LLM")
    
    cache_status: CacheStatus = Field(
        ...,
        description="Whether response came from cache"
    )
    
    sources: Optional[List[SourceChunk]] = Field(
        default=None,
        description="Retrieved source chunks (if include_sources=True)"
    )
    
    cache_similarity: Optional[float] = Field(
        default=None,
        description="Similarity score if cache hit"
    )
    
    processing_time_ms: float = Field(
        ...,
        description="Time taken to process request in milliseconds"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Response timestamp"
    )
    
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional response metadata"
    )


class DocumentUploadResponse(BaseModel):
    """
    Response after successful document upload.
    """
    document_id: str = Field(..., description="Unique document identifier")
    
    filename: str = Field(..., description="Original filename")
    
    document_type: DocumentType = Field(..., description="Detected document type")
    
    num_chunks: int = Field(..., description="Number of chunks created")
    
    total_characters: int = Field(..., description="Total characters processed")
    
    processing_time_ms: float = Field(..., description="Processing time")
    
    status: str = Field(default="success", description="Upload status")
    
    message: str = Field(
        default="Document processed and indexed successfully",
        description="Status message"
    )


# ============================================================
# INTERNAL MODELS - Used between services (not exposed in API)
# ============================================================

class DocumentChunk(BaseModel):
    """
    Internal representation of a document chunk.
    
    This is what we store in vector database.
    """
    chunk_id: str = Field(..., description="Unique chunk identifier")
    
    content: str = Field(..., description="Chunk text content")
    
    embedding: Optional[List[float]] = Field(
        default=None,
        description="Vector embedding (384 or 768 dimensions)"
    )
    # This is the actual vector - list of floats
    
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Chunk metadata"
    )
    """
    Metadata examples:
    {
        "document_id": "doc_123",
        "filename": "ml_paper.pdf",
        "page": 5,
        "chunk_index": 12,
        "document_type": "pdf",
        "created_at": "2024-02-08T10:30:00"
    }
    """
    
    class Config:
        # Allow arbitrary types (needed for numpy arrays if we use them)
        arbitrary_types_allowed = True


class CacheEntry(BaseModel):
    """
    Semantic cache entry.
    
    Stored in Redis: {query_embedding: CacheEntry}
    """
    query: str = Field(..., description="Original query text")
    
    query_embedding: List[float] = Field(
        ...,
        description="Query embedding vector"
    )
    
    response: str = Field(..., description="Cached response")
    
    sources: Optional[List[SourceChunk]] = Field(
        default=None,
        description="Cached source chunks"
    )
    
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Cache entry creation time"
    )
    
    hit_count: int = Field(
        default=0,
        description="Number of times this cache entry was used"
    )
    # Useful for analytics - which queries are most common?
    
    ttl: Optional[int] = Field(
        default=3600,  # 1 hour in seconds
        description="Time to live in seconds"
    )
    # Auto-delete cache entries after 1 hour (configurable)


# ============================================================
# UTILITY MODELS
# ============================================================

class HealthCheck(BaseModel):
    """Health check response for monitoring"""
    status: str = Field(default="healthy", description="Service status")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: str = Field(default="1.0.0", description="API version")
    services: Dict[str, str] = Field(
        default_factory=dict,
        description="Status of dependent services"
    )
    """
    Example:
    {
        "status": "healthy",
        "timestamp": "2024-02-08T10:30:00",
        "version": "1.0.0",
        "services": {
            "vector_db": "connected",
            "cache": "connected",
            "llm": "available"
        }
    }
    """


class ErrorResponse(BaseModel):
    """Standard error response format"""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    detail: Optional[str] = Field(default=None, description="Detailed error info")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        # Example for API documentation
        schema_extra = {
            "example": {
                "error": "ValidationError",
                "message": "Query cannot be empty",
                "detail": "Field 'query' failed validation",
                "timestamp": "2024-02-08T10:30:00"
            }
        }