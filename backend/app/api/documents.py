"""
Document management API endpoints.
Fully tenant-isolated: all queries and file operations are scoped to the user's tenant.
"""
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from fastapi.responses import Response
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.database import get_db
from app.models.user import User
from app.models.document import Document, DocumentStatus, DocumentType
from app.api.auth import get_current_user
from app.middleware.file_storage import get_storage
from app.vector_store.store import TenantVectorStore
from app.vector_store.hybrid_search import HybridSearcher

settings = get_settings()

router = APIRouter()

# Allowed MIME types → DocumentType mapping
MIME_TYPE_MAP = {
    "application/pdf": DocumentType.PDF,
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": DocumentType.DOCX,
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": DocumentType.PPTX,
    "application/vnd.ms-powerpoint": DocumentType.PPTX,
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": DocumentType.XLSX,
    "image/png": DocumentType.IMAGE,
    "image/jpeg": DocumentType.IMAGE,
    "image/tiff": DocumentType.IMAGE,
    "text/csv": DocumentType.CSV,
    "application/csv": DocumentType.CSV,
    "text/plain": DocumentType.OTHER,
}

MAX_FILE_SIZE = settings.max_file_size_mb * 1024 * 1024


@router.get("/")
async def list_documents(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List all documents for current tenant."""
    result = await db.execute(
        select(Document)
        .where(Document.tenant_id == current_user.tenant_id)
        .order_by(Document.created_at.desc())
    )
    documents = result.scalars().all()
    return {"documents": documents}


@router.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Upload a new document.
    Files are stored via the configured storage backend (local or GCS).
    Processing is queued as a background Celery task.
    """
    # 1. Validate file type
    content_type = file.content_type or "application/octet-stream"
    doc_type = MIME_TYPE_MAP.get(content_type)
    if doc_type is None:
        allowed = ", ".join(MIME_TYPE_MAP.keys())
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type: {content_type}. Allowed: {allowed}",
        )

    # 2. Read file content and check size
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds {settings.max_file_size_mb}MB limit",
        )

    # 3. Save file via storage backend (local disk or GCS)
    storage = get_storage()
    original_name = file.filename or "unnamed_file"
    stored_path = storage.save(
        tenant_id=current_user.tenant_id,
        filename=original_name,
        content=content,
    )

    # 4. Create database record
    document = Document(
        filename=original_name,
        original_filename=original_name,
        file_path=stored_path,
        file_size=len(content),
        mime_type=content_type,
        document_type=doc_type,
        status=DocumentStatus.PENDING,
        tenant_id=current_user.tenant_id,
        owner_id=current_user.id,
    )
    db.add(document)
    await db.flush()
    await db.refresh(document)

    # Commit so the row is visible to the Celery worker's separate DB session
    await db.commit()

    # 5. Queue background processing task
    try:
        from app.celery_app import celery_app  # noqa: F401 — ensures Redis broker is initialized
        from app.worker import process_document

        process_document.delay(str(document.id))
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Failed to queue document processing task: {e}")
        # We don't raise here to allow the upload to succeed, but we log it.
        # Ideally, we might want to set status to FAILED or similar.

    return {
        "id": str(document.id),
        "filename": document.original_filename,
        "status": document.status.value,
        "file_size": document.file_size,
        "document_type": document.document_type.value,
        "message": "Document uploaded successfully. Processing started.",
    }


@router.get("/{document_id}")
async def get_document(
    document_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get a specific document. Only returns documents belonging to the user's tenant."""
    result = await db.execute(
        select(Document)
        .where(Document.id == document_id)
        .where(Document.tenant_id == current_user.tenant_id)
    )
    document = result.scalar_one_or_none()

    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    return {
        "id": str(document.id),
        "filename": document.original_filename,
        "status": document.status.value,
        "file_size": document.file_size,
        "document_type": document.document_type.value,
        "chunk_count": document.chunk_count,
        "created_at": document.created_at.isoformat() if document.created_at else None,
        "indexed_at": document.indexed_at.isoformat() if document.indexed_at else None,
        "error_message": document.error_message,
    }


@router.get("/{document_id}/download")
async def download_document(
    document_id: UUID,
    inline: bool = False,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Serve the raw file for a document.
    ?inline=false (default) → Content-Disposition: attachment  (browser download)
    ?inline=true            → Content-Disposition: inline      (open in browser tab)
    """
    result = await db.execute(
        select(Document)
        .where(Document.id == document_id)
        .where(Document.tenant_id == current_user.tenant_id)
    )
    document = result.scalar_one_or_none()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    if not document.file_path:
        raise HTTPException(status_code=404, detail="File not available")

    storage = get_storage()
    try:
        content = storage.read(current_user.tenant_id, document.file_path)
    except Exception:
        raise HTTPException(status_code=404, detail="File not found in storage")

    if inline:
        disposition = "inline"
    else:
        safe_name = document.original_filename.replace('"', '_')
        disposition = f'attachment; filename="{safe_name}"'

    return Response(
        content=content,
        media_type=document.mime_type or "application/octet-stream",
        headers={"Content-Disposition": disposition},
    )


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Delete a document. Only deletes documents belonging to the user's tenant."""
    result = await db.execute(
        select(Document)
        .where(Document.id == document_id)
        .where(Document.tenant_id == current_user.tenant_id)
    )
    document = result.scalar_one_or_none()

    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    # Delete vectors from Qdrant
    vector_store = TenantVectorStore()
    vector_store.delete_document_vectors(str(current_user.tenant_id), str(document_id))

    # Invalidate BM25 cache so stale chunks don't appear in hybrid search
    hybrid_searcher = HybridSearcher(vector_store=vector_store)
    hybrid_searcher.invalidate_cache(str(current_user.tenant_id))

    # Delete the file from storage backend
    if document.file_path:
        storage = get_storage()
        storage.delete(current_user.tenant_id, document.file_path)

    await db.delete(document)
    return None

