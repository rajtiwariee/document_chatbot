"""
Embedding generation using Google Gemini's text-embedding-004 model.

Generates vector embeddings for document chunks to enable semantic search.
"""
import logging
from typing import Optional

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.config import get_settings
from app.document_processing.chunker import Chunk

settings = get_settings()
logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """
    Generate embeddings for text chunks using Google's text-embedding-004.

    The embedding model produces 768-dimensional vectors.
    """

    DIMENSIONS = 768  # text-embedding-004 output size

    def __init__(self, api_key: Optional[str] = None):
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=f"models/{settings.embedding_model}",
            google_api_key=api_key or settings.google_api_key,
        )

    def embed_text(self, text: str) -> list[float]:
        """Generate embedding for a single text string."""
        return self.embeddings.embed_query(text)

    def embed_chunks(self, chunks: list[Chunk]) -> list[list[float]]:
        """
        Generate embeddings for a list of chunks.

        Uses batch embedding for efficiency.
        Returns list of embedding vectors in the same order as chunks.
        """
        if not chunks:
            return []

        texts = [chunk.text for chunk in chunks]
        embeddings = self.embeddings.embed_documents(texts)

        logger.info(
            f"Generated {len(embeddings)} embeddings "
            f"(dim={len(embeddings[0]) if embeddings else 0})"
        )
        return embeddings

    def embed_query(self, query: str) -> list[float]:
        """
        Generate embedding for a search query.

        Note: Some embedding models use different encodings for
        queries vs documents. This method ensures the correct
        query encoding is used.
        """
        return self.embeddings.embed_query(query)
