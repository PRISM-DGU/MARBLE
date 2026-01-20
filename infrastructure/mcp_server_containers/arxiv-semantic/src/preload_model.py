"""
Pre-load sentence transformers model to avoid runtime delays and rate limiting.
This script should be run during container initialization.
"""

import os
import logging
import time
from sentence_transformers import SentenceTransformer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_cache_environment():
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

def preload_model(model_name: str = "all-MiniLM-L6-v2", max_retries: int = 3):
    """
    Pre-load the sentence transformer model with retry logic.

    Args:
        model_name: Name of the model to preload
        max_retries: Maximum number of retry attempts
    """
    setup_cache_environment()

    cache_dir = os.environ.get('TRANSFORMERS_CACHE', '/workspace/.cache/transformers')

    for attempt in range(max_retries):
        try:
            logger.info(f"Pre-loading model: {model_name} (attempt {attempt + 1}/{max_retries})")

            # Load the model - this will download it if not cached
            model = SentenceTransformer(
                model_name,
                cache_folder=cache_dir
            )

            # Test the model with a simple encoding to ensure it's working
            test_text = "Test sentence for model verification"
            test_embedding = model.encode([test_text])

            logger.info(f"✅ Model {model_name} loaded successfully!")
            logger.info(f"   Cache directory: {cache_dir}")
            logger.info(f"   Test embedding shape: {test_embedding.shape}")

            # Clean up to free memory
            del model
            return True

        except Exception as e:
            logger.error(f"❌ Failed to load model on attempt {attempt + 1}: {e}")

            if attempt < max_retries - 1:
                # Exponential backoff with jitter
                delay = (2 ** attempt) + 1
                logger.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                logger.error(f"Failed to load model after {max_retries} attempts")
                return False

def verify_model_cache(model_name: str = "all-MiniLM-L6-v2"):
    """Verify that the model is properly cached."""
    cache_dir = os.environ.get('TRANSFORMERS_CACHE', '/workspace/.cache/transformers')

    # Check for model files
    model_path = os.path.join(cache_dir, f"sentence-transformers_{model_name}")

    if os.path.exists(model_path):
        logger.info(f"✅ Model cache verified at: {model_path}")
        # List some files to confirm
        try:
            files = os.listdir(model_path)
            logger.info(f"   Cached files: {files[:5]}{'...' if len(files) > 5 else ''}")
        except Exception as e:
            logger.warning(f"Could not list cache files: {e}")
        return True
    else:
        logger.warning(f"⚠️ Model cache not found at: {model_path}")
        return False

if __name__ == "__main__":
    logger.info("🚀 Starting model pre-loading process...")

    # Pre-load the default model
    success = preload_model()

    if success:
        # Verify the cache
        verify_model_cache()
        logger.info("🎉 Model pre-loading completed successfully!")
    else:
        logger.error("💥 Model pre-loading failed!")
        exit(1)