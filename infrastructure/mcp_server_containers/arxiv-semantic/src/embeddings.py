"""
Embeddings module for semantic search using sentence transformers.
Based on Zotero MCP's approach but optimized for ArXiv papers.
"""

import os
import logging
import time
import random
from typing import List, Optional, Dict, Any
from sentence_transformers import SentenceTransformer
import numpy as np
import torch
from functools import wraps

logger = logging.getLogger(__name__)


def retry_with_exponential_backoff(max_retries: int = 5, base_delay: float = 1.0):
    """Decorator for exponential backoff retry logic."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise

                    # Calculate delay with jitter
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.2f}s...")
                    time.sleep(delay)
            return None
        return wrapper
    return decorator


class EmbeddingManager:
    """Manages embeddings generation using sentence transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding manager.

        Args:
            model_name: Name of the sentence transformer model to use
        """
        self.model_name = model_name
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._setup_cache_environment()

        # Circuit breaker state
        self._failure_count = 0
        self._last_failure_time = None
        self._circuit_breaker_threshold = 3
        self._circuit_breaker_timeout = 300  # 5 minutes

        self._initialize_model()

    def _setup_cache_environment(self):
        """Set up proper caching environment variables."""
        # Set persistent cache directory
        cache_root = os.environ.get('HF_HOME', '/workspace/.cache/huggingface')

        # Set all relevant cache environment variables
        os.environ['HF_HOME'] = cache_root
        os.environ['HF_HUB_CACHE'] = os.path.join(cache_root, 'hub')
        os.environ['TRANSFORMERS_CACHE'] = os.path.join(cache_root, 'transformers')

        # Create cache directories if they don't exist
        for cache_dir in [cache_root, os.environ['HF_HUB_CACHE'], os.environ['TRANSFORMERS_CACHE']]:
            os.makedirs(cache_dir, exist_ok=True)

        logger.info(f"Cache environment set up - HF_HOME: {cache_root}")

    def _is_circuit_breaker_open(self) -> bool:
        """Check if circuit breaker is open (preventing operations)."""
        if self._failure_count < self._circuit_breaker_threshold:
            return False

        if self._last_failure_time is None:
            return False

        # Check if timeout has passed
        current_time = time.time()
        if current_time - self._last_failure_time > self._circuit_breaker_timeout:
            logger.info("Circuit breaker timeout passed, attempting to reset")
            self._failure_count = 0
            self._last_failure_time = None
            return False

        return True

    def _record_failure(self):
        """Record a failure for circuit breaker logic."""
        self._failure_count += 1
        self._last_failure_time = time.time()
        logger.warning(f"Circuit breaker failure count: {self._failure_count}")

    def _record_success(self):
        """Record a success, resetting circuit breaker if needed."""
        if self._failure_count > 0:
            logger.info("Operation successful, resetting circuit breaker")
            self._failure_count = 0
            self._last_failure_time = None

    def is_healthy(self) -> bool:
        """Check if the embedding manager is healthy and ready."""
        if self._is_circuit_breaker_open():
            logger.warning("Circuit breaker is open - service unavailable")
            return False

        return self.model is not None

    @retry_with_exponential_backoff(max_retries=5, base_delay=2.0)
    def _initialize_model(self):
        """Initialize the sentence transformer model with retry logic."""
        if self._is_circuit_breaker_open():
            raise RuntimeError("Circuit breaker is open - model loading suspended")

        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            cache_dir = os.environ.get('TRANSFORMERS_CACHE', '/workspace/.cache/transformers')

            # Check if model is already cached locally
            model_cache_path = os.path.join(cache_dir, f"sentence-transformers_{self.model_name}")
            if os.path.exists(model_cache_path):
                logger.info(f"Using cached model from: {model_cache_path}")
            else:
                logger.info(f"Model not cached, downloading from Hugging Face Hub...")

            self.model = SentenceTransformer(
                self.model_name,
                cache_folder=cache_dir,
                device=self.device
            )
            logger.info(f"Model loaded successfully on {self.device}")
            self._record_success()
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            self._record_failure()
            raise

    def generate_embeddings(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of texts to generate embeddings for
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar

        Returns:
            numpy array of embeddings
        """
        if not texts:
            return np.array([])

        if not self.is_healthy():
            raise RuntimeError("EmbeddingManager is not healthy - circuit breaker may be open")

        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True
            )
            self._record_success()
            return embeddings
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            self._record_failure()
            raise

    def generate_query_embedding(self, query: str) -> np.ndarray:
        """
        Generate embedding for a single query.

        Args:
            query: Query text

        Returns:
            numpy array of embedding
        """
        return self.generate_embeddings([query], show_progress=False)[0]

    def compute_similarity(
        self,
        query_embedding: np.ndarray,
        document_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Compute cosine similarity between query and documents.

        Args:
            query_embedding: Query embedding vector
            document_embeddings: Document embedding matrix

        Returns:
            Array of similarity scores
        """
        # Normalize embeddings
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        doc_norms = document_embeddings / np.linalg.norm(
            document_embeddings, axis=1, keepdims=True
        )

        # Compute cosine similarity
        similarities = np.dot(doc_norms, query_norm)
        return similarities

    def chunk_text(
        self,
        text: str,
        chunk_size: int = 512,
        overlap: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Split text into overlapping chunks for embedding.

        Args:
            text: Text to chunk
            chunk_size: Maximum chunk size in tokens (approximate)
            overlap: Number of tokens to overlap between chunks

        Returns:
            List of chunk dictionaries with text and metadata
        """
        # Simple character-based chunking (can be improved with tokenizer)
        chunks = []
        chars_per_token = 4  # Approximate
        chunk_chars = chunk_size * chars_per_token
        overlap_chars = overlap * chars_per_token

        start = 0
        chunk_id = 0

        while start < len(text):
            end = min(start + chunk_chars, len(text))
            chunk_text = text[start:end]

            # Find last complete sentence if possible
            if end < len(text):
                last_period = chunk_text.rfind('. ')
                if last_period > chunk_chars // 2:
                    end = start + last_period + 1
                    chunk_text = text[start:end]

            chunks.append({
                'id': chunk_id,
                'text': chunk_text.strip(),
                'start': start,
                'end': end,
                'length': end - start
            })

            chunk_id += 1
            start = end - overlap_chars if end < len(text) else end

        return chunks