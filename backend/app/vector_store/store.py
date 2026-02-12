"""
Multi-tenant Qdrant vector store wrapper.

Each tenant gets a separate Qdrant collection for strong data isolation.
Collection naming: tenant_{tenant_id}
"""
import logging
from uuid import UUID
from dataclasses import dataclass, field

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)

from app.config import get_settings
from app.document_processing.chunker import Chunk

settings = get_settings()
logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A single search result from vector store."""
    chunk_text: str
    document_id: str
    page_number: int | None
    score: float
    metadata: dict = field(default_factory=dict)


class TenantVectorStore:
    """
    Multi-tenant Qdrant vector store.

    Each tenant gets an isolated collection. All operations are scoped
    to a specific tenant's collection to prevent data leakage.
    """

    def __init__(self):
        self.vector_size = settings.embedding_dimensions
        self.client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            timeout=10,
        )

    def _collection_name(self, tenant_id: str) -> str:
        """Generate collection name for a tenant."""
        return f"tenant_{tenant_id}"

    def get_or_create_collection(self, tenant_id: str) -> str:
        """
        Ensure a tenant-specific collection exists.
        Creates one if it doesn't exist.
        """
        collection_name = self._collection_name(tenant_id)

        try:
            collections = self.client.get_collections().collections
            existing_names = [c.name for c in collections]

            if collection_name not in existing_names:
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE,
                    ),
                )
                logger.info(f"Created Qdrant collection: {collection_name}")
            return collection_name
        except Exception as e:
            logger.error(f"Error creating collection {collection_name}: {e}")
            raise

    def add_chunks(
        self,
        tenant_id: str,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> int:
        """
        Add document chunks with their embeddings to the tenant's collection.

        Returns the number of points upserted.
        """
        if not chunks or not embeddings:
            return 0

        collection_name = self.get_or_create_collection(tenant_id)

        points = [
            PointStruct(
                id=chunk.id,
                vector=embedding,
                payload={
                    "text": chunk.text,
                    "document_id": chunk.document_id,
                    "tenant_id": chunk.tenant_id,
                    "page_number": chunk.page_number,
                    "chunk_index": chunk.chunk_index,
                    "source_file": chunk.metadata.get("source_file", ""),
                    "file_type": chunk.metadata.get("file_type", ""),
                },
            )
            for chunk, embedding in zip(chunks, embeddings)
        ]

        # Upsert in batches of 100
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self.client.upsert(
                collection_name=collection_name,
                points=batch,
            )

        logger.info(
            f"Upserted {len(points)} vectors to {collection_name}"
        )
        return len(points)

    def search(
        self,
        tenant_id: str,
        query_embedding: list[float],
        top_k: int = 5,
        document_id: str | None = None,
    ) -> list[SearchResult]:
        """
        Search within a tenant's documents.

        Args:
            tenant_id: Tenant to search within
            query_embedding: Query vector
            top_k: Number of results to return
            document_id: Optional filter to search within a specific document

        Returns:
            List of SearchResult ordered by relevance score
        """
        collection_name = self._collection_name(tenant_id)

        # Check if collection exists
        try:
            self.client.get_collection(collection_name)
        except Exception:
            logger.warning(f"Collection {collection_name} not found")
            return []

        # Build filter
        query_filter = None
        if document_id:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id),
                    )
                ]
            )

        response = self.client.query_points(
            collection_name=collection_name,
            query=query_embedding,
            query_filter=query_filter,
            limit=top_k,
        )

        return [
            SearchResult(
                chunk_text=hit.payload.get("text", ""),
                document_id=hit.payload.get("document_id", ""),
                page_number=hit.payload.get("page_number"),
                score=hit.score,
                metadata={
                    "source_file": hit.payload.get("source_file", ""),
                    "file_type": hit.payload.get("file_type", ""),
                    "chunk_index": hit.payload.get("chunk_index", 0),
                },
            )
            for hit in response.points
        ]

    def delete_document_vectors(self, tenant_id: str, document_id: str) -> None:
        """
        Delete all vectors for a specific document within a tenant's collection.
        Called when a document is deleted.
        """
        collection_name = self._collection_name(tenant_id)

        try:
            self.client.delete(
                collection_name=collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=MatchValue(value=document_id),
                        )
                    ]
                ),
            )
            logger.info(
                f"Deleted vectors for document {document_id} "
                f"from {collection_name}"
            )
        except Exception as e:
            logger.error(f"Error deleting vectors: {e}")

    def delete_tenant_collection(self, tenant_id: str) -> None:
        """Delete an entire tenant's collection. Use with extreme caution."""
        collection_name = self._collection_name(tenant_id)
        try:
            self.client.delete_collection(collection_name)
            logger.info(f"Deleted collection: {collection_name}")
        except Exception as e:
            logger.error(f"Error deleting collection {collection_name}: {e}")
