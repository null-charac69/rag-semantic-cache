"""
Semantic Cache Service
======================
Implements intelligent caching based on semantic similarity.

Traditional cache:
- Exact match only
- "What is ML?" ≠ "What is machine learning?" (cache miss!)

Semantic cache:
- Similarity-based matching
- "What is ML?" ≈ "What is machine learning?" (cache hit!)
- Saves LLM calls and improves speed

Strategy:
1. User asks question
2. Embed question → vector
3. Check if similar question exists in cache (similarity >= threshold)
4. If yes: return cached answer (FAST!)
5. If no: run full RAG pipeline, cache result
"""

import redis
import json
from typing import Optional, List, Dict, Any, Union
from datetime import datetime, timedelta
from loguru import logger
import pickle
import numpy as np

from app.core.config import settings
from app.models.schemas import CacheEntry, CacheStatus, SourceChunk
from app.services.embedding import get_embedding_service


class SemanticCache:
    """
    Semantic cache implementation using Redis.
    
    Why Redis?
    - In-memory = FAST (microsecond lookups)
    - Supports TTL (auto-expire old entries)
    - Atomic operations (no race conditions)
    - Easy to deploy
    
    Storage strategy:
    - Key: cache_entry:{hash}
    - Value: Pickled CacheEntry object
    - Index: Separate key for query embeddings (for similarity search)
    """
    
    def __init__(self):
        """Initialize Redis connection and embedding service."""
        logger.info("Initializing semantic cache...")
        
        try:
            # Connect to Redis
            self.redis_client = redis.from_url(
                settings.REDIS_URL,
                decode_responses=False  # We'll store binary (pickle)
            )
            
            # Test connection
            self.redis_client.ping()
            
            # Get embedding service
            self.embedding_service = get_embedding_service()
            
            # Cache key prefix
            self.cache_prefix = "rag_cache:"
            self.embedding_index_key = f"{self.cache_prefix}embeddings_index"
            
            logger.success("✅ Semantic cache initialized")
            
        except redis.ConnectionError as e:
            logger.warning(
                f"⚠️ Redis connection failed ({settings.REDIS_URL}): {e}. "
                f"Switching to in-memory semantic cache."
            )
            raise
        except Exception as e:
            logger.error(f"❌ Failed to initialize cache: {e}")
            raise
    
    
    def _generate_cache_key(self, query_hash: str) -> str:
        """Generate Redis key for cache entry."""
        return f"{self.cache_prefix}entry:{query_hash}"
    
    
    def _hash_query(self, query: str) -> str:
        """
        Generate hash for query.
        
        Used as cache key identifier.
        """
        import hashlib
        return hashlib.md5(query.encode()).hexdigest()[:16]
    
    
    def check_cache(
        self, 
        query: str,
        threshold: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Check if similar query exists in cache.
        
        Process:
        1. Embed query
        2. Get all cached query embeddings
        3. Calculate similarity with each
        4. If any similarity >= threshold: CACHE HIT!
        5. Return cached response
        
        Args:
            query: User's question
            threshold: Similarity threshold (default from config)
            
        Returns:
            Dict with cached response and metadata, or None if miss
        """
        if not settings.CACHE_ENABLED or not self.redis_client:
            return None
        
        threshold = threshold or settings.CACHE_SIMILARITY_THRESHOLD
        
        try:
            logger.info(f"Checking cache for: '{query[:50]}...'")
            
            # Step 1: Embed query
            query_embedding = self.embedding_service.embed_text(query)
            
            # Step 2: Get cached embeddings index
            # This is a JSON list of {hash, embedding} pairs
            embeddings_data = self.redis_client.get(self.embedding_index_key)
            
            if not embeddings_data:
                logger.info("Cache empty (no embeddings index)")
                return None
            
            cached_embeddings = json.loads(embeddings_data)
            if not cached_embeddings:
                return None
            
            # Step 3: Vectorized batch cosine similarity computation (NumPy)
            matrix = np.array([item['embedding'] for item in cached_embeddings], dtype=np.float32)
            q_vec = np.array(query_embedding, dtype=np.float32)
            
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1e-9
            q_norm = np.linalg.norm(q_vec)
            if q_norm == 0:
                q_norm = 1e-9
                
            sims = np.dot(matrix / norms, q_vec / q_norm)
            best_idx = int(np.argmax(sims))
            best_similarity = float(sims[best_idx])
            best_match = cached_embeddings[best_idx]['hash']
            
            # Step 4: Check if best match exceeds threshold
            if best_similarity >= threshold:
                logger.info(
                    f"🎯 CACHE HIT! Similarity: {best_similarity:.3f} "
                    f"(threshold: {threshold})"
                )
                
                # Retrieve cached entry
                cache_key = self._generate_cache_key(best_match)
                cached_data = self.redis_client.get(cache_key)
                
                if cached_data:
                    cache_entry = pickle.loads(cached_data)
                    
                    # Increment hit count
                    cache_entry.hit_count += 1
                    self._update_cache_entry(best_match, cache_entry)
                    
                    return {
                        "response": cache_entry.response,
                        "sources": cache_entry.sources,
                        "cached_query": cache_entry.query,
                        "similarity": best_similarity,
                        "hit_count": cache_entry.hit_count,
                        "created_at": cache_entry.created_at
                    }
            
            logger.info(
                f"❌ CACHE MISS. Best similarity: {best_similarity:.3f} "
                f"< threshold: {threshold}"
            )
            return None
            
        except Exception as e:
            logger.error(f"Error checking cache: {e}")
            return None
    
    
    def store_in_cache(
        self,
        query: str,
        response: str,
        sources: Optional[List[SourceChunk]] = None,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Store query-response pair in cache.
        
        Process:
        1. Embed query
        2. Create CacheEntry object
        3. Store in Redis with TTL
        4. Update embeddings index
        
        Args:
            query: User's question
            response: Generated answer
            sources: Retrieved source chunks
            ttl: Time to live in seconds (default 1 hour)
            
        Returns:
            True if successful
        """
        if not settings.CACHE_ENABLED or not self.redis_client:
            return False
        
        ttl = ttl or 3600  # 1 hour default
        
        try:
            logger.info(f"Storing in cache: '{query[:50]}...'")
            
            # Step 1: Embed query
            query_embedding = self.embedding_service.embed_text(query)
            
            # Step 2: Create cache entry
            cache_entry = CacheEntry(
                query=query,
                query_embedding=query_embedding,
                response=response,
                sources=sources,
                created_at=datetime.utcnow(),
                hit_count=0,
                ttl=ttl
            )
            
            # Step 3: Generate hash and key
            query_hash = self._hash_query(query)
            cache_key = self._generate_cache_key(query_hash)
            
            # Step 4: Store entry in Redis
            cache_data = pickle.dumps(cache_entry)
            self.redis_client.setex(
                cache_key,
                ttl,
                cache_data
            )
            
            # Step 5: Update embeddings index
            self._update_embeddings_index(query_hash, query_embedding)
            
            logger.success(f"✅ Stored in cache (TTL: {ttl}s)")
            return True
            
        except Exception as e:
            logger.error(f"Error storing in cache: {e}")
            return False
    
    
    def _update_embeddings_index(
        self,
        query_hash: str,
        embedding: List[float]
    ) -> None:
        """
        Update the embeddings index.
        
        The index is a JSON list of all cached query embeddings.
        Used for similarity search during cache lookup.
        
        Structure:
        [
            {"hash": "a3f5b2c8", "embedding": [0.2, -0.4, ...]},
            {"hash": "b7d9e1f3", "embedding": [0.3, -0.3, ...]},
            ...
        ]
        """
        try:
            # Get current index
            embeddings_data = self.redis_client.get(self.embedding_index_key)
            
            if embeddings_data:
                embeddings_list = json.loads(embeddings_data)
            else:
                embeddings_list = []
            
            # Check if hash already exists (update scenario)
            hash_exists = False
            for item in embeddings_list:
                if item['hash'] == query_hash:
                    item['embedding'] = embedding
                    hash_exists = True
                    break
            
            # Add new entry if doesn't exist
            if not hash_exists:
                embeddings_list.append({
                    "hash": query_hash,
                    "embedding": embedding
                })
            
            # Store updated index
            self.redis_client.set(
                self.embedding_index_key,
                json.dumps(embeddings_list)
            )
            
        except Exception as e:
            logger.error(f"Error updating embeddings index: {e}")
    
    
    def _update_cache_entry(
        self,
        query_hash: str,
        cache_entry: CacheEntry
    ) -> None:
        """
        Update existing cache entry (e.g., increment hit count).
        
        Maintains original TTL.
        """
        try:
            cache_key = self._generate_cache_key(query_hash)
            
            # Get remaining TTL
            ttl = self.redis_client.ttl(cache_key)
            if ttl <= 0:
                ttl = cache_entry.ttl
            
            # Store updated entry
            cache_data = pickle.dumps(cache_entry)
            self.redis_client.setex(cache_key, ttl, cache_data)
            
        except Exception as e:
            logger.error(f"Error updating cache entry: {e}")
    
    
    def invalidate_cache(self, query: Optional[str] = None) -> int:
        """
        Invalidate cache entries.
        
        Args:
            query: Specific query to invalidate (None = clear all)
            
        Returns:
            Number of entries invalidated
        """
        if not self.redis_client:
            return 0
        
        try:
            if query:
                # Invalidate specific query
                query_hash = self._hash_query(query)
                cache_key = self._generate_cache_key(query_hash)
                deleted = self.redis_client.delete(cache_key)
                
                logger.info(f"Invalidated cache for: '{query}'")
                return deleted
            else:
                # Clear entire cache
                keys = self.redis_client.keys(f"{self.cache_prefix}*")
                if keys:
                    deleted = self.redis_client.delete(*keys)
                    logger.warning(f"⚠️ Cleared entire cache: {deleted} entries")
                    return deleted
                return 0
                
        except Exception as e:
            logger.error(f"Error invalidating cache: {e}")
            return 0
    
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dict with cache metrics
        """
        if not self.redis_client:
            return {"enabled": False}
        
        try:
            # Count cache entries
            keys = self.redis_client.keys(f"{self.cache_prefix}entry:*")
            
            # Get embeddings index size
            embeddings_data = self.redis_client.get(self.embedding_index_key)
            embeddings_count = 0
            if embeddings_data:
                embeddings_count = len(json.loads(embeddings_data))
            
            return {
                "enabled": True,
                "total_entries": len(keys),
                "embeddings_indexed": embeddings_count,
                "threshold": settings.CACHE_SIMILARITY_THRESHOLD,
                "redis_url": settings.REDIS_URL
            }
            
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {"enabled": True, "error": str(e)}


# ============================================================
# MOCK CACHE (Fallback when Redis unavailable)
# ============================================================

class MockCache:
    """
    In-memory semantic cache fallback.
    
    Used when Redis is not available or for local testing.
    Uses vectorized NumPy similarity searches and tracks entry TTLs.
    """
    
    def __init__(self):
        self.cache_store: Dict[str, Any] = {}
        self.embeddings_store: List[Dict[str, Any]] = []
        self.embedding_service = get_embedding_service()
        logger.info("⚡ In-memory semantic cache active (vectorized fallback)")
    
    def check_cache(
        self,
        query: str,
        threshold: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """Check in-memory semantic cache using vectorized cosine similarity."""
        if not settings.CACHE_ENABLED or not self.embeddings_store:
            return None
        
        threshold = threshold or settings.CACHE_SIMILARITY_THRESHOLD
        now = datetime.utcnow()
        
        # Filter expired entries
        valid_items = []
        for item in self.embeddings_store:
            h = item['hash']
            entry = self.cache_store.get(h)
            if entry:
                expires_at = entry.get('expires_at')
                if expires_at and now > expires_at:
                    self.cache_store.pop(h, None)
                    continue
                valid_items.append(item)
        self.embeddings_store = valid_items
        
        if not self.embeddings_store:
            return None
            
        query_embedding = self.embedding_service.embed_text(query)
        
        # Vectorized similarity matching
        matrix = np.array([item['embedding'] for item in self.embeddings_store], dtype=np.float32)
        q_vec = np.array(query_embedding, dtype=np.float32)
        
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9
        q_norm = np.linalg.norm(q_vec)
        if q_norm == 0:
            q_norm = 1e-9
            
        sims = np.dot(matrix / norms, q_vec / q_norm)
        best_idx = int(np.argmax(sims))
        best_similarity = float(sims[best_idx])
        best_match = self.embeddings_store[best_idx]['hash']
        
        if best_similarity >= threshold and best_match in self.cache_store:
            logger.info(f"🎯 IN-MEMORY CACHE HIT! Similarity: {best_similarity:.3f} >= {threshold}")
            entry = self.cache_store[best_match]
            entry['hit_count'] += 1
            entry['similarity'] = best_similarity
            return entry
        
        return None
    
    def store_in_cache(
        self,
        query: str,
        response: str,
        sources: Optional[List[SourceChunk]] = None,
        ttl: Optional[int] = None
    ) -> bool:
        """Store in mock cache with TTL support."""
        if not settings.CACHE_ENABLED:
            return False
        
        import hashlib
        query_hash = hashlib.md5(query.encode()).hexdigest()[:16]
        query_embedding = self.embedding_service.embed_text(query)
        ttl = ttl or 3600
        now = datetime.utcnow()
        
        self.cache_store[query_hash] = {
            "response": response,
            "sources": sources,
            "cached_query": query,
            "similarity": 1.0,
            "hit_count": 0,
            "created_at": now,
            "expires_at": now + timedelta(seconds=ttl)
        }
        
        # Update or append
        existing = next((i for i in self.embeddings_store if i['hash'] == query_hash), None)
        if existing:
            existing['embedding'] = query_embedding
        else:
            self.embeddings_store.append({
                "hash": query_hash,
                "embedding": query_embedding
            })
        
        logger.info(f"✅ Stored in in-memory cache (hash={query_hash}, TTL={ttl}s)")
        return True
    
    def invalidate_cache(self, query: Optional[str] = None) -> int:
        """Invalidate mock cache."""
        if query is None:
            count = len(self.cache_store)
            self.cache_store.clear()
            self.embeddings_store.clear()
            return count
        return 0
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get mock cache stats."""
        return {
            "enabled": True,
            "total_entries": len(self.cache_store),
            "type": "mock (in-memory)"
        }


# ============================================================
# SINGLETON INSTANCE
# ============================================================

_cache_instance = None

def get_cache() -> SemanticCache:
    """
    Get or create singleton cache.
    
    Falls back to MockCache if Redis unavailable.
    
    Usage:
        from app.services.cache import get_cache
        
        cache = get_cache()
        result = cache.check_cache("What is AI?")
    """
    global _cache_instance
    if _cache_instance is None:
        try:
            _cache_instance = SemanticCache()
        except Exception as e:
            logger.warning(f"Falling back to mock cache: {e}")
            _cache_instance = MockCache()
    return _cache_instance
