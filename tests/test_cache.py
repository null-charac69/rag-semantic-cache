"""
Unit tests for the Semantic Caching layer.
Verifies exact match, semantic similarity thresholding, and cache invalidation.
"""

import pytest
import numpy as np
from datetime import datetime
from app.services.cache import MockCache
from app.models.schemas import SourceChunk


class DummyEmbeddingService:
    """Mock embedding service for deterministic offline testing."""
    def embed_text(self, text: str):
        # Deterministic mock vectors: words encode into low-dimensional space
        text_lower = text.lower()
        if "machine learning" in text_lower or "ml" in text_lower:
            return [0.9, 0.1, 0.0, 0.0]
        elif "deep learning" in text_lower:
            return [0.85, 0.15, 0.0, 0.0]
        elif "pizza" in text_lower or "food" in text_lower:
            return [0.0, 0.0, 0.9, 0.1]
        else:
            return [0.5, 0.5, 0.5, 0.5]


@pytest.fixture
def test_cache(monkeypatch):
    cache = MockCache()
    cache.embedding_service = DummyEmbeddingService()
    return cache


def test_cache_miss_on_empty(test_cache):
    result = test_cache.check_cache("What is machine learning?")
    assert result is None


def test_cache_hit_similar_phrasing(test_cache):
    # Store ML answer
    test_cache.store_in_cache(
        query="What is machine learning?",
        response="Machine learning is a subset of artificial intelligence."
    )
    
    # Query with slightly different phrasing that matches dummy embedding
    hit = test_cache.check_cache("Explain ML concept", threshold=0.85)
    assert hit is not None
    assert "subset of artificial intelligence" in hit["response"]
    assert hit["hit_count"] == 1


def test_cache_miss_different_topic(test_cache):
    test_cache.store_in_cache(
        query="What is machine learning?",
        response="ML is AI."
    )
    
    # Query unrelated domain
    miss = test_cache.check_cache("How to make a pizza?", threshold=0.85)
    assert miss is None


def test_cache_invalidation(test_cache):
    test_cache.store_in_cache("Query 1", "Answer 1")
    assert test_cache.get_cache_stats()["total_entries"] == 1
    
    cleared = test_cache.invalidate_cache()
    assert cleared == 1
    assert test_cache.get_cache_stats()["total_entries"] == 0
