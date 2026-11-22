"""
Embedding Service
=================
Converts text into vector embeddings using sentence-transformers.

What are embeddings?
- Text → Numbers (vectors)
- Similar text → Similar vectors
- Enables semantic search (meaning-based, not keyword matching)

Example:
"What is ML?" → [0.23, -0.45, 0.67, ...] (384 numbers)
"Machine learning?" → [0.24, -0.44, 0.65, ...] (very similar!)
"Pizza recipe" → [-0.89, 0.12, -0.34, ...] (very different)
"""

from sentence_transformers import SentenceTransformer
from typing import List, Union
import numpy as np
from loguru import logger
from functools import lru_cache

from app.core.config import settings


class EmbeddingService:
    """
    Handles all embedding operations.
    
    Why a class?
    - Model loaded once, reused many times (expensive to load!)
    - Encapsulates all embedding logic
    - Easy to swap models later
    """
    
    def __init__(self):
        """
        Initialize the embedding model.
        
        This happens once at app startup.
        Model is ~80MB, takes 2-3 seconds to load.
        """
        logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
        
        try:
            self.model = SentenceTransformer(settings.EMBEDDING_MODEL)
            
            # Get embedding dimension (needed for vector DB setup)
            self.dimension = self.model.get_sentence_embedding_dimension()
            
            logger.success(
                f"✅ Embedding model loaded successfully "
                f"(dimension: {self.dimension})"
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to load embedding model: {e}")
            raise
    
    
    def embed_text(self, text: str) -> List[float]:
        """
        Convert single text string to embedding vector.
        
        Args:
            text: Input text (query or document chunk)
            
        Returns:
            List of floats (embedding vector)
            
        Example:
            >>> service = EmbeddingService()
            >>> vec = service.embed_text("Hello world")
            >>> len(vec)
            384
            >>> type(vec[0])
            float
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding")
            # Return zero vector for empty text
            return [0.0] * self.dimension
        
        try:
            # encode() does the magic: text → vector
            embedding = self.model.encode(
                text,
                convert_to_numpy=True,  # Return as numpy array
                show_progress_bar=False  # No progress bar for single text
            )
            
            # Convert numpy array to Python list (for JSON serialization)
            return embedding.tolist()
            
        except Exception as e:
            logger.error(f"Error embedding text: {e}")
            raise
    
    
    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Convert multiple texts to embeddings efficiently.
        
        Why batch processing?
        - Much faster than one-by-one (uses GPU/CPU parallelization)
        - When uploading document with 100 chunks, batch them!
        
        Args:
            texts: List of text strings
            batch_size: How many to process at once (32 is good balance)
            
        Returns:
            List of embedding vectors
            
        Example:
            >>> texts = ["Hello", "World", "AI is amazing"]
            >>> embeddings = service.embed_batch(texts)
            >>> len(embeddings)
            3
            >>> len(embeddings[0])
            384
        """
        if not texts:
            logger.warning("Empty text list provided for batch embedding")
            return []
        
        # Filter out empty strings
        valid_texts = [t for t in texts if t and t.strip()]
        
        if not valid_texts:
            logger.warning("All texts were empty after filtering")
            return [[0.0] * self.dimension] * len(texts)
        
        try:
            logger.info(f"Embedding batch of {len(valid_texts)} texts...")
            
            # Batch encoding - THE POWER MOVE
            embeddings = self.model.encode(
                valid_texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=len(valid_texts) > 50  # Show bar if many texts
            )
            
            logger.success(f"✅ Batch embedded {len(valid_texts)} texts")
            
            # Convert to list of lists
            return embeddings.tolist()
            
        except Exception as e:
            logger.error(f"Error in batch embedding: {e}")
            raise
    
    
    def compute_similarity(
        self, 
        embedding1: List[float], 
        embedding2: List[float]
    ) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Cosine similarity formula:
        similarity = (A · B) / (||A|| * ||B||)
        
        Result range: -1 to 1
        - 1.0 = identical vectors
        - 0.0 = orthogonal (unrelated)
        - -1.0 = opposite vectors
        
        Args:
            embedding1: First vector
            embedding2: Second vector
            
        Returns:
            Similarity score (0-1, we normalize to positive range)
            
        Example:
            >>> vec1 = service.embed_text("machine learning")
            >>> vec2 = service.embed_text("artificial intelligence")
            >>> similarity = service.compute_similarity(vec1, vec2)
            >>> print(similarity)
            0.87  # High similarity!
        """
        try:
            # Convert to numpy for fast computation
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Cosine similarity calculation
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            
            # Normalize to 0-1 range (some models can give negative)
            # In practice, sentence-transformers usually gives 0-1
            return float(max(0.0, min(1.0, similarity)))
            
        except Exception as e:
            logger.error(f"Error computing similarity: {e}")
            return 0.0
    
    
    def find_most_similar(
        self,
        query_embedding: List[float],
        candidate_embeddings: List[List[float]],
        top_k: int = 3
    ) -> List[tuple[int, float]]:
        """
        Find top-k most similar embeddings from candidates.
        
        Used for semantic cache lookup!
        
        Args:
            query_embedding: The query vector
            candidate_embeddings: List of cached query vectors
            top_k: How many to return
            
        Returns:
            List of (index, similarity_score) tuples, sorted by score
            
        Example:
            >>> query_vec = service.embed_text("What is AI?")
            >>> cached_vecs = [
            ...     service.embed_text("Machine learning basics"),
            ...     service.embed_text("What is artificial intelligence?"),
            ...     service.embed_text("Pizza recipe")
            ... ]
            >>> results = service.find_most_similar(query_vec, cached_vecs, top_k=2)
            >>> print(results)
            [(1, 0.95), (0, 0.72)]  # Index 1 (AI question) is most similar!
        """
        if not candidate_embeddings:
            return []
        
        try:
            # Calculate similarity with all candidates
            similarities = [
                (idx, self.compute_similarity(query_embedding, candidate))
                for idx, candidate in enumerate(candidate_embeddings)
            ]
            
            # Sort by similarity score (descending)
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Return top-k
            return similarities[:top_k]
            
        except Exception as e:
            logger.error(f"Error finding most similar: {e}")
            return []
    
    
    def get_dimension(self) -> int:
        """
        Get embedding dimension.
        
        Needed when initializing vector database.
        """
        return self.dimension
    
    
    def get_model_name(self) -> str:
        """Get the name of the embedding model being used."""
        return settings.EMBEDDING_MODEL


# ============================================================
# SINGLETON INSTANCE
# ============================================================

@lru_cache(maxsize=1)
def get_embedding_service() -> EmbeddingService:
    """
    Get or create singleton embedding service.
    
    Why singleton?
    - Model is expensive to load (80MB, 2-3 seconds)
    - Load once, reuse everywhere
    - Memory efficient
    
    @lru_cache ensures this function returns the SAME instance every time.
    
    Usage in other files:
        from app.services.embedding import get_embedding_service
        
        embedding_service = get_embedding_service()
        vector = embedding_service.embed_text("Hello world")
    """
    return EmbeddingService()


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def normalize_text(text: str) -> str:
    """
    Normalize text before embedding.
    
    Why normalize?
    - Consistent embeddings
    - Remove noise
    - Better similarity matching
    
    Steps:
    1. Strip whitespace
    2. Replace multiple spaces with single space
    3. Lowercase (optional, depends on model)
    """
    if not text:
        return ""
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    # Replace multiple spaces/newlines with single space
    import re
    text = re.sub(r'\s+', ' ', text)
    
    return text


def chunk_text_for_embedding(text: str, max_length: int = 512) -> List[str]:
    """
    Split long text into chunks for embedding.
    
    Why?
    - Most embedding models have max token limit (512 tokens ≈ 2000 chars)
    - Long text → truncated → loss of information
    - Better to split and embed separately
    
    Args:
        text: Long text to split
        max_length: Max characters per chunk
        
    Returns:
        List of text chunks
    """
    if len(text) <= max_length:
        return [text]
    
    chunks = []
    words = text.split()
    current_chunk = []
    current_length = 0
    
    for word in words:
        word_length = len(word) + 1  # +1 for space
        
        if current_length + word_length > max_length:
            # Current chunk is full, start new one
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_length = word_length
        else:
            current_chunk.append(word)
            current_length += word_length
    
    # Add last chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks
