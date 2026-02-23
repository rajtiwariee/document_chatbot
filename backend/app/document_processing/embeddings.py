"""
Embedding generation using Google Gemini embedding models.

Generates vector embeddings for document chunks to enable semantic search.
Batches large requests to stay within API limits.
"""
import logging
import time
from typing import Optional

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.config import get_settings
from app.document_processing.chunker import Chunk

settings = get_settings()
logger = logging.getLogger(__name__)

# Gemini embedding API accepts up to 100 texts per batch request
EMBED_BATCH_SIZE = 100


class EmbeddingGenerator:
    """
    Generate embeddings for text chunks using the configured embedding model.

    Vector dimensions are configured via settings.embedding_dimensions.
    """

    def __init__(self, api_key: Optional[str] = None):
        model_name = settings.embedding_model
        if not model_name.startswith("models/"):
            model_name = f"models/{model_name}"

        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=model_name,
            google_api_key=api_key or settings.google_api_key,
        )

    def embed_text(self, text: str) -> list[float]:
        """Generate embedding for a single text string."""
        return self.embeddings.embed_query(text)

    def embed_chunks(self, chunks: list[Chunk]) -> list[list[float]]:
        """
        Generate embeddings for a list of chunks.

        Batches in groups of EMBED_BATCH_SIZE to stay within API limits.
        Includes retry with backoff for transient failures.
        """
        if not chunks:
            return []

        texts = [chunk.text for chunk in chunks]
        all_embeddings: list[list[float]] = []

        total_batches = (len(texts) + EMBED_BATCH_SIZE - 1) // EMBED_BATCH_SIZE

        for batch_idx in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = texts[batch_idx:batch_idx + EMBED_BATCH_SIZE]
            batch_num = batch_idx // EMBED_BATCH_SIZE + 1

            logger.info(
                f"Embedding batch {batch_num}/{total_batches} "
                f"({len(batch)} texts)"
            )

            # Retry with exponential backoff
            for attempt in range(3):
                try:
                    embeddings = self.embeddings.embed_documents(batch)
                    all_embeddings.extend(embeddings)
                    break
                except Exception as e:
                    if attempt < 2:
                        wait = 2 ** (attempt + 1)
                        logger.warning(
                            f"Embedding batch {batch_num} failed (attempt {attempt + 1}): {e}. "
                            f"Retrying in {wait}s..."
                        )
                        time.sleep(wait)
                    else:
                        logger.error(f"Embedding batch {batch_num} failed after 3 attempts: {e}")
                        raise

        logger.info(
            f"Generated {len(all_embeddings)} embeddings "
            f"(dim={len(all_embeddings[0]) if all_embeddings else 0})"
        )
        return all_embeddings

    def embed_query(self, query: str) -> list[float]:
        """
        Generate embedding for a search query.

        Note: Some embedding models use different encodings for
        queries vs documents. This method ensures the correct
        query encoding is used.
        """
        return self.embeddings.embed_query(query)
