"""
Document management API endpoints.
Fully tenant-isolated: all queries and file operations are scoped to the user's tenant.
"""
import os
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.database import get_db
from app.models.user import User
from app.models.document import Document, DocumentStatus, DocumentType
from app.api.auth import get_current_user
from app.middleware.tenant import get_tenant_id, require_same_tenant
from app.middleware.file_storage import generate_storage_path, validate_file_access

settings = get_settings()

router = APIRouter()

# Allowed MIME types → DocumentType mapping
MIME_TYPE_MAP = {
    "application/pdf": DocumentType.PDF,
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": DocumentType.DOCX,
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": DocumentType.PPTX,
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": DocumentType.XLSX,
    "image/png": DocumentType.IMAGE,
    "image/jpeg": DocumentType.IMAGE,
    "image/tiff": DocumentType.IMAGE,
}

MAX_FILE_SIZE = settings.max_file_size_mb * 1024 * 1024  # Convert MB to bytes


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
    Files are stored in a tenant-isolated directory.
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

    # 3. Generate tenant-isolated storage path
    stored_filename, full_path = generate_storage_path(
        tenant_id=current_user.tenant_id,
        original_filename=file.filename or "unnamed_file",
    )

    # 4. Write file to disk
    with open(full_path, "wb") as f:
        f.write(content)

    # 5. Create database record
    document = Document(
        filename=stored_filename,
        original_filename=file.filename or "unnamed_file",
        file_path=full_path,
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

    # 6. Queue background processing task
    try:
        from app.worker import process_document

        process_document.delay(str(document.id))
    except Exception:
        # If Celery is unavailable, document stays in PENDING status
        # Users can trigger reprocessing later
        pass

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

    # Delete the file from disk (tenant-isolated path)
    if document.file_path and os.path.exists(document.file_path):
        # Double-check the file belongs to this tenant's directory
        if validate_file_access(current_user.tenant_id, document.file_path):
            os.remove(document.file_path)

    await db.delete(document)
    return None
