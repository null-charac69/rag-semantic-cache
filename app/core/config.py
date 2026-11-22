"""
Configuration Management
========================
This file handles all app configuration using Pydantic Settings.
Why Pydantic? It validates environment variables at startup and gives type safety.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional
import os


class Settings(BaseSettings):
    """
    Application Settings
    
    How this works:
    1. Pydantic automatically reads from .env file
    2. Validates types (str, int, float, bool)
    3. Provides defaults if values aren't set
    4. Raises error if required fields are missing
    """
    
    # ============================================================
    # API KEYS - Required for external services
    # ============================================================
    GROQ_API_KEY: str = Field(
        default="",
        description="Groq API key for LLM"
    )
    # Recommended for production LLM generation (free at https://console.groq.com)
    
    # ============================================================
    # MODEL CONFIGURATION
    # ============================================================
    EMBEDDING_MODEL: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence transformer model for embeddings"
    )
    # This is a 384-dimensional model, fast and efficient
    # Alternatives: all-mpnet-base-v2 (768-dim, more accurate but slower)
    
    LLM_MODEL: str = Field(
        default="openai/gpt-oss-120b",
        description="Groq LLM model name"
    )
    # Groq: openai/gpt-oss-120b (default). Also: llama-3.1-70b, mixtral-8x7b, gemma-7b
    
    # ============================================================
    # VECTOR DATABASE SETTINGS
    # ============================================================
    CHROMA_PERSIST_DIR: str = Field(
        default="./data/vector_db",
        description="Directory to persist Chroma database"
    )
    # Chroma stores vectors on disk here - survives restarts
    
    COLLECTION_NAME: str = Field(
        default="rag_documents",
        description="Chroma collection name"
    )
    # Think of this as a "table" in SQL - one collection per use case
    
    # ============================================================
    # SEMANTIC CACHE CONFIGURATION
    # ============================================================
    CACHE_SIMILARITY_THRESHOLD: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Cosine similarity threshold for cache hits"
    )
    """
    How semantic cache works:
    - User asks: "What is machine learning?"
    - We compute embedding of this query
    - Check if any cached query has similarity >= 0.95
    - If yes: return cached response (fast!)
    - If no: run full RAG pipeline (slower)
    
    Threshold tuning:
    - 0.95-1.0: Very strict (only near-identical queries match)
    - 0.85-0.95: Moderate (similar phrasing matches)
    - 0.70-0.85: Loose (semantically similar matches)
    """
    
    CACHE_ENABLED: bool = Field(
        default=True,
        description="Enable/disable semantic caching"
    )
    
    REDIS_URL: str = Field(
        default="redis://localhost:6379",
        description="Redis connection URL"
    )
    # Redis stores: {query_embedding: response} pairs
    # Super fast in-memory lookup
    
    # ============================================================
    # CHUNKING CONFIGURATION
    # ============================================================
    CHUNK_SIZE: int = Field(
        default=1000,
        description="Text chunk size in characters"
    )
    # Why chunk? LLMs have context limits. We break documents into pieces.
    # 1000 chars ≈ 250 tokens (rough estimate)
    
    CHUNK_OVERLAP: int = Field(
        default=200,
        description="Overlap between chunks"
    )
    # Overlap prevents losing context at chunk boundaries
    # Example: "...end of chunk 1. Start of..." becomes
    # Chunk 1: "...end of chunk 1."
    # Chunk 2: "chunk 1. Start of..."
    
    # ============================================================
    # RAG RETRIEVAL SETTINGS
    # ============================================================
    TOP_K_RESULTS: int = Field(
        default=3,
        description="Number of relevant chunks to retrieve"
    )
    # We fetch top 3 most similar chunks and pass to LLM
    # More chunks = more context but slower + more tokens
    
    # ============================================================
    # APPLICATION SETTINGS
    # ============================================================
    DEBUG: bool = Field(default=False, description="Debug mode")
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    
    MAX_FILE_SIZE_MB: int = Field(
        default=10,
        description="Maximum upload file size in MB"
    )
    
    # ============================================================
    # MODEL CONFIG - Load from .env
    # ============================================================
    class Config:
        env_file = ".env"  # Load from .env file
        case_sensitive = True  # GROQ_API_KEY != groq_api_key


# ============================================================
# SINGLETON INSTANCE
# ============================================================
settings = Settings()
"""
Why singleton?
- We only need ONE config object for entire app
- Load once at startup, use everywhere
- Memory efficient

Usage in other files:
    from app.core.config import settings
    api_key = settings.GROQ_API_KEY
"""


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def ensure_directories():
    """
    Create necessary directories if they don't exist.
    Called at app startup.
    """
    directories = [
        settings.CHROMA_PERSIST_DIR,
        "./data/documents",
        "./data/cache",
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        # exist_ok=True means "don't error if already exists"
    
    print(f"✅ Directories initialized: {', '.join(directories)}")


def validate_config(strict: bool = False):
    """
    Validate configuration at startup.
    Warns or fails fast if configuration is incomplete.
    """
    # Check if API key is set
    if not settings.GROQ_API_KEY or settings.GROQ_API_KEY == "your_groq_api_key_here":
        msg = "⚠️ GROQ_API_KEY not set! LLM generation will require a valid key in .env (get one at https://console.groq.com/keys)."
        if strict:
            raise ValueError(msg)
        print(msg)
    
    # Validate thresholds
    if not 0.0 <= settings.CACHE_SIMILARITY_THRESHOLD <= 1.0:
        raise ValueError("❌ CACHE_SIMILARITY_THRESHOLD must be between 0 and 1")
    
    print("✅ Configuration validated successfully")
    print(f"   - Embedding Model: {settings.EMBEDDING_MODEL}")
    print(f"   - LLM Model: {settings.LLM_MODEL}")
    print(f"   - Cache Enabled: {settings.CACHE_ENABLED}")
    print(f"   - Cache Threshold: {settings.CACHE_SIMILARITY_THRESHOLD}")