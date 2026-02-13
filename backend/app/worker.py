"""
Celery worker tasks for document processing.

Full pipeline: Extract → Chunk → Embed → Index → Notify

Each task handles its own error reporting and status updates.
"""
import logging
from datetime import datetime

from celery import shared_task
from sqlalchemy import create_engine, update
from sqlalchemy.orm import Session

from app.config import get_settings
from app.models.document import Document, DocumentStatus

settings = get_settings()
logger = logging.getLogger(__name__)

# Sync database engine for Celery workers (Celery uses sync workers)
# Convert async URL to sync: postgresql+asyncpg → postgresql+psycopg2
SYNC_DB_URL = settings.database_url.replace("+asyncpg", "+psycopg2")


def get_sync_session() -> Session:
    """Create a sync DB session for Celery tasks."""
    engine = create_engine(SYNC_DB_URL)
    return Session(engine)


def update_document_status(
    document_id: str,
    status: DocumentStatus,
    error_message: str | None = None,
    chunk_count: int | None = None,
    indexed_at: datetime | None = None,
):
    """Update document status in the database."""
    session = get_sync_session()
    try:
        values = {"status": status}
        if error_message is not None:
            values["error_message"] = error_message
        if chunk_count is not None:
            values["chunk_count"] = chunk_count
        if indexed_at is not None:
            values["indexed_at"] = indexed_at

        session.execute(
            update(Document)
            .where(Document.id == document_id)
            .values(**values)
        )
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Failed to update document status: {e}")
    finally:
        session.close()


def get_document(document_id: str) -> dict | None:
    """Fetch document metadata from the database."""
    session = get_sync_session()
    try:
        doc = session.query(Document).filter(Document.id == document_id).first()
        if doc:
            return {
                "id": str(doc.id),
                "file_path": doc.file_path,
                "file_type": doc.document_type.value if doc.document_type else "other",
                "tenant_id": str(doc.tenant_id),
                "owner_id": str(doc.owner_id),
                "filename": doc.original_filename,
            }
        return None
    finally:
        session.close()


@shared_task(
    name="process_document",
    bind=True,
    max_retries=3,
    default_retry_delay=30,
    acks_late=True,
)
def process_document(self, document_id: str):
    """
    Full document processing pipeline:
    1. Extract text from document
    2. Chunk text into retrieval-sized pieces
    3. Generate embeddings via Google Gemini
    4. Store vectors in Qdrant

    This task is idempotent — re-running will overwrite previous results.
    """
    logger.info(f"Starting document processing: {document_id}")

    # ── Step 0: Fetch document metadata ──────────────────────────────
    doc = get_document(document_id)
    if not doc:
        logger.warning(
            f"Document not found: {document_id} (attempt {self.request.retries + 1})"
        )
        raise self.retry(
            exc=ValueError(f"Document not found: {document_id}"),
            countdown=2 ** (self.request.retries + 1),  # 2s, 4s, 8s
        )

    # Update status to processing
    update_document_status(document_id, DocumentStatus.PROCESSING)

    try:
        # ── Step 1: Read file from storage backend ───────────────────
        logger.info(f"[{document_id}] Reading file from storage...")

        import tempfile
        import os
        from uuid import UUID
        from app.middleware.file_storage import get_storage

        storage = get_storage()
        tenant_uuid = UUID(doc["tenant_id"])

        # For local storage, file_path is already a local path
        # For GCS, we download to a temp file
        if settings.storage_backend == "gcs":
            file_content = storage.read(tenant_uuid, doc["file_path"])
            _, ext = os.path.splitext(doc["filename"])
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
            tmp.write(file_content)
            tmp.close()
            local_path = tmp.name
        else:
            local_path = doc["file_path"]

        # ── Step 2: Extract text ─────────────────────────────────────
        logger.info(f"[{document_id}] Extracting text from {doc['filename']}")

        from app.document_processing.extractor import DocumentExtractor

        extractor = DocumentExtractor()
        extracted = extractor.extract(local_path, doc["file_type"])

        # Clean up temp file if we created one
        if settings.storage_backend == "gcs":
            os.unlink(local_path)

        if extracted.error:
            raise RuntimeError(f"Extraction failed: {extracted.error}")

        if not extracted.full_text().strip():
            raise RuntimeError("No text could be extracted from document")

        logger.info(
            f"[{document_id}] Extracted {extracted.page_count} pages, "
            f"{len(extracted.full_text())} chars"
        )

        # ── Step 2: Chunk text ───────────────────────────────────────
        logger.info(f"[{document_id}] Chunking text...")

        from app.document_processing.chunker import SemanticChunker

        chunker = SemanticChunker(chunk_size=1000, chunk_overlap=200)
        chunks = chunker.chunk(extracted, document_id, doc["tenant_id"])

        if not chunks:
            raise RuntimeError("No chunks produced from document text")

        logger.info(f"[{document_id}] Created {len(chunks)} chunks")

        # ── Step 3: Generate embeddings ──────────────────────────────
        logger.info(f"[{document_id}] Generating embeddings...")

        from app.document_processing.embeddings import EmbeddingGenerator

        generator = EmbeddingGenerator()
        embeddings = generator.embed_chunks(chunks)

        logger.info(f"[{document_id}] Generated {len(embeddings)} embeddings")

        # ── Step 4: Store in vector database ─────────────────────────
        logger.info(f"[{document_id}] Storing vectors in Qdrant...")

        from app.vector_store.store import TenantVectorStore

        vector_store = TenantVectorStore()
        num_stored = vector_store.add_chunks(doc["tenant_id"], chunks, embeddings)

        logger.info(f"[{document_id}] Stored {num_stored} vectors")

        # ── Step 4b: Invalidate BM25 cache ───────────────────────────
        try:
            from app.vector_store.hybrid_search import HybridSearcher
            hybrid_searcher = HybridSearcher(vector_store=vector_store)
            hybrid_searcher.invalidate_cache(doc["tenant_id"])
        except Exception as e:
            logger.warning(f"[{document_id}] Failed to invalidate BM25 cache: {e}")

        # ── Step 5: Mark as completed ────────────────────────────────
        update_document_status(
            document_id,
            DocumentStatus.COMPLETED,
            chunk_count=len(chunks),
            indexed_at=datetime.utcnow(),
        )

        logger.info(f"Document processing complete: {document_id}")

        return {
            "status": "completed",
            "document_id": document_id,
            "pages": extracted.page_count,
            "chunks": len(chunks),
            "vectors": num_stored,
        }

    except Exception as exc:
        logger.exception(f"Document processing failed: {document_id}")

        # Mark as failed in DB
        update_document_status(
            document_id,
            DocumentStatus.FAILED,
            error_message=str(exc),
        )

        # Retry on transient errors (up to max_retries)
        if self.request.retries < self.max_retries:
            raise self.retry(exc=exc)

        return {
            "status": "failed",
            "document_id": document_id,
            "error": str(exc),
        }
