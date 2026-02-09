"""
Document management API endpoints.
"""
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.user import User
from app.models.document import Document, DocumentStatus
from app.api.auth import get_current_user

router = APIRouter()


@router.get("/")
async def list_documents(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
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
    db: AsyncSession = Depends(get_db)
):
    """Upload a new document."""
    # TODO: Implement file upload and processing
    return {
        "message": "Document upload endpoint - implementation pending",
        "filename": file.filename,
        "user_id": str(current_user.id),
        "tenant_id": str(current_user.tenant_id)
    }


@router.get("/{document_id}")
async def get_document(
    document_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get a specific document."""
    result = await db.execute(
        select(Document)
        .where(Document.id == document_id)
        .where(Document.tenant_id == current_user.tenant_id)
    )
    document = result.scalar_one_or_none()
    
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return document


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Delete a document."""
    result = await db.execute(
        select(Document)
        .where(Document.id == document_id)
        .where(Document.tenant_id == current_user.tenant_id)
    )
    document = result.scalar_one_or_none()
    
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    await db.delete(document)
    return None
