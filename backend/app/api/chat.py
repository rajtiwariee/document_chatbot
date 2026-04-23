"""
Chat API endpoints.

Provides endpoints for chatting with the LangGraph ReAct agent,
streaming responses, and managing conversation history.
Supports both JSON and multipart/form-data (with file attachments).
"""
import base64 as _b64
import json
import logging
import os
import re
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile
from starlette.datastructures import UploadFile as StarletteUploadFile
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from langchain_core.messages import HumanMessage, AIMessage

from app.config import get_settings
from app.database import get_db
from app.middleware.file_storage import get_storage
from app.models.user import User
from app.models.conversation import Conversation, Message
from app.api.auth import get_current_user
from app.agent.graph import create_agent_graph
from app.agent.attachment_processor import (
    process_attachments,
    AttachmentContext,
    ALLOWED_MIMES,
)

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter()


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str
    conversation_id: str | None = None


class SourceCitation(BaseModel):
    document: str
    page: int | None = None
    relevance: float | None = None


class ChatResponse(BaseModel):
    message: str
    sources: list[dict] = []
    conversation_id: str


class ConversationSummary(BaseModel):
    id: str
    title: str
    created_at: str
    message_count: int


# ---------------------------------------------------------------------------
# Attachment validation
# ---------------------------------------------------------------------------
def _validate_files(files: list[UploadFile]) -> None:
    """Validate attachment count, size, and MIME type."""
    max_count = settings.chat_attachment_max_count
    max_bytes = settings.chat_attachment_max_size_mb * 1024 * 1024

    if len(files) > max_count:
        raise HTTPException(
            status_code=400,
            detail=f"Too many files. Maximum {max_count} attachments allowed.",
        )

    for f in files:
        mime = f.content_type or "application/octet-stream"
        if mime not in ALLOWED_MIMES:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {mime} ({f.filename})",
            )
        if f.size and f.size > max_bytes:
            raise HTTPException(
                status_code=400,
                detail=f"File too large: {f.filename} ({f.size / 1024 / 1024:.1f}MB, max {settings.chat_attachment_max_size_mb}MB)",
            )


# ---------------------------------------------------------------------------
# Build multimodal HumanMessage from attachments
# ---------------------------------------------------------------------------
def _build_human_message(
    text: str,
    attachment_ctx: AttachmentContext | None,
) -> HumanMessage:
    """Build a HumanMessage, potentially with multimodal content blocks."""
    if not attachment_ctx or not attachment_ctx.attachments:
        return HumanMessage(content=text or "")

    content_blocks = []

    # Collect text context from all attachment types (images via Qwen VLM get text descriptions here)
    extra_context_parts = []
    for att in attachment_ctx.attachments:
        if att.category == "document" and att.extracted_text:
            extra_context_parts.append(
                f"\n--- Attached file: {att.filename} ---\n{att.extracted_text}"
            )
        elif att.category == "spreadsheet" and att.spreadsheet_summary:
            extra_context_parts.append(
                f"\n--- Attached spreadsheet: {att.filename} (attachment_id: {att.attachment_id}) ---\n"
                f"{att.spreadsheet_summary}"
            )
        elif att.category == "image" and att.extracted_text:
            # Qwen VLM processed: pass text description only — no raw image bytes to Gemini
            extra_context_parts.append(
                f"\n--- Attached image: {att.filename} ---\n"
                f"[Visual content extracted by vision model]\n{att.extracted_text}"
            )

    full_text = (text or "").strip()
    if extra_context_parts:
        full_text += "\n" + "\n".join(extra_context_parts)

    if full_text:
        content_blocks.append({"type": "text", "text": full_text})

    # Add raw image_url blocks (Gemini backend only — images not pre-processed by Qwen VLM)
    for att in attachment_ctx.attachments:
        if att.category == "image" and not att.extracted_text and att.image_data_url:
            content_blocks.append({
                "type": "image_url",
                "image_url": {"url": att.image_data_url},
            })

    if not content_blocks:
        return HumanMessage(content=text or "")

    block_summary = [
        f"{b.get('type')}({len(b.get('image_url', {}).get('url', ''))}chars)"
        if b.get("type") == "image_url"
        else b.get("type")
        for b in content_blocks
        if isinstance(b, dict)
    ]
    logger.info(f"Built multimodal HumanMessage with {len(content_blocks)} blocks: {block_summary}")
    return HumanMessage(content=content_blocks)


def _build_attachment_db_records(
    ctx: AttachmentContext, tenant_id: uuid.UUID
) -> list[dict]:
    """
    Persist chat attachments and return JSONB-ready metadata.
    - local backend: stores base64 bytes inline in the returned dict
    - gcs backend: uploads to GCS, stores blob path in the returned dict
    """
    if not ctx or not ctx.attachments:
        return []

    settings = get_settings()
    is_gcs = getattr(settings, "storage_backend", "local") == "gcs"
    storage = get_storage() if is_gcs else None

    records = []
    for att in ctx.attachments:
        temp_path = os.path.join(ctx.temp_dir, f"{att.attachment_id}_{att.filename}")
        try:
            with open(temp_path, "rb") as f:
                raw_bytes = f.read()

            record = {
                "id": att.attachment_id,
                "filename": att.filename,
                "mime_type": att.mime_type or "application/octet-stream",
            }

            if is_gcs:
                gcs_path = storage.save(tenant_id, att.filename, raw_bytes)
                record["gcs_path"] = gcs_path
            else:
                record["content_b64"] = _b64.b64encode(raw_bytes).decode("utf-8")

            records.append(record)
        except Exception as e:
            logger.error(f"Failed to persist attachment {att.filename}: {e}")

    return records


def _serialize_attachments_for_response(
    attachments: list | None, message_id: str
) -> list[dict]:
    """
    Returns attachment metadata shaped for the frontend.
    - local: embeds base64 data URL directly (no extra fetch needed)
    - gcs: returns URL to the backend proxy endpoint
    """
    if not attachments:
        return []
    result = []
    for att in attachments:
        mime = att.get("mime_type", "")
        is_image = mime.startswith("image/")
        entry = {
            "id": att["id"],
            "filename": att["filename"],
            "mime_type": mime,
        }
        if is_image:
            if "content_b64" in att:
                entry["data_url"] = f"data:{mime};base64,{att['content_b64']}"
            elif "gcs_path" in att:
                entry["url"] = (
                    f"/api/chat/attachments"
                    f"?message_id={message_id}&attachment_id={att['id']}"
                )
        result.append(entry)
    return result


def _get_attachment_ids(attachment_ctx: AttachmentContext | None) -> list[str]:
    """Extract spreadsheet attachment IDs for agent state."""
    if not attachment_ctx:
        return []
    return [
        a.attachment_id
        for a in attachment_ctx.attachments
        if a.category == "spreadsheet"
    ]


# ---------------------------------------------------------------------------
# Chat Endpoints
# ---------------------------------------------------------------------------
@router.post("/", response_model=ChatResponse)
async def chat_endpoint(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Send a message to the document assistant.
    Accepts JSON or multipart/form-data (with file attachments).
    """
    content_type = request.headers.get("content-type", "")
    if "multipart/form-data" in content_type:
        form = await request.form()
        message = form.get("message", "")
        conversation_id = form.get("conversation_id") or None
        raw_files = form.getlist("files")
        files = [f for f in raw_files if isinstance(f, StarletteUploadFile)]
        logger.info(f"Chat: {len(files)} file(s) attached — {[(f.filename, f.content_type) for f in files]}")
    else:
        body = await request.json()
        message = body.get("message", "")
        conversation_id = body.get("conversation_id")
        files = []

    return await _handle_chat(message, conversation_id, files, current_user, db)


@router.post("/stream")
async def chat_stream(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Stream chat response via Server-Sent Events (SSE).
    Accepts JSON or multipart/form-data (with file attachments).
    """
    content_type = request.headers.get("content-type", "")
    if "multipart/form-data" in content_type:
        form = await request.form()
        message = form.get("message", "")
        conversation_id = form.get("conversation_id") or None
        raw_files = form.getlist("files")
        files = [f for f in raw_files if isinstance(f, StarletteUploadFile)]
        logger.info(f"[stream] Chat: {len(files)} file(s) attached — {[(f.filename, f.content_type) for f in files]}")
    else:
        body = await request.json()
        message = body.get("message", "")
        conversation_id = body.get("conversation_id")
        files = []

    return await _handle_chat_stream(message, conversation_id, files, current_user, db)


# ---------------------------------------------------------------------------
# Core chat logic
# ---------------------------------------------------------------------------
async def _handle_chat(
    message: str,
    conversation_id: str | None,
    files: list[UploadFile],
    current_user: User,
    db: AsyncSession,
) -> ChatResponse:
    """Core chat handler shared by JSON and multipart endpoints."""
    tenant_id = str(current_user.tenant_id)
    user_id = str(current_user.id)

    # Validate files
    if files:
        _validate_files(files)

    attachment_ctx = None
    try:
        # Process attachments for immediate LLM context
        if files:
            attachment_ctx = await process_attachments(files)
            
            # --- START NEW BACKGROUND INDEXING LOGIC (Option A) ---
            # Automatically push document/spreadsheet attachments to the formal 
            # Knowledge Base vector pipeline so they are remembered in the future.
            
            try:
                from app.api.documents import MIME_TYPE_MAP
                from app.models.document import Document, DocumentStatus, DocumentType
                
                # We need to seek backend-stored files or reuse the temp files. 
                # Since we already saved them to `ctx.temp_dir`, we can read from there.
                storage = get_storage()
                for att in attachment_ctx.attachments:
                    # Only index dense documents/spreadsheets in Vector DB, ignore standalone tiny images
                    if att.category in ["document", "spreadsheet"]:
                        temp_path = os.path.join(attachment_ctx.temp_dir, f"{att.attachment_id}_{att.filename}")
                        with open(temp_path, "rb") as f:
                            raw_bytes = f.read()
                        
                        # Save to permanent storage engine (GCS/Local)
                        stored_path = storage.save(current_user.tenant_id, att.filename, raw_bytes)
                        doc_type = MIME_TYPE_MAP.get(att.mime_type, DocumentType.OTHER)
                        
                        # Create formal Database record
                        document = Document(
                            filename=att.filename,
                            original_filename=att.filename,
                            file_path=stored_path,
                            file_size=len(raw_bytes),
                            mime_type=att.mime_type,
                            document_type=doc_type,
                            status=DocumentStatus.PENDING,
                            tenant_id=current_user.tenant_id,
                            owner_id=current_user.id,
                        )
                        db.add(document)
                        await db.flush()
                        
                        # Queue in Celery background worker
                        try:
                            from app.celery_app import celery_app
                            from app.worker import process_document
                            process_document.delay(str(document.id))
                            logger.info(f"Queued chat attachment {att.filename} for permanent Vector DB storage.")
                        except Exception as ce:
                            logger.error(f"Failed to queue chat attachment for Celery: {ce}")
                            
            except Exception as e:
                logger.error(f"Failed to push chat attachments to Knowledge Base pipeline: {e}")
            # --- END NEW BACKGROUND INDEXING LOGIC ---

        # Get or create conversation
        conversation, history_messages = await _get_or_create_conversation(
            db, current_user, conversation_id, message,
        )

        # Save user message
        msg_count = len(history_messages)
        user_msg = Message(
            conversation_id=conversation.id,
            role="user",
            content=message,
            sequence=msg_count,
        )
        db.add(user_msg)
        await db.flush()
        await db.refresh(user_msg)

        if attachment_ctx:
            records = _build_attachment_db_records(attachment_ctx, current_user.tenant_id)
            user_msg.attachments = records or None

        # Build agent state
        human_msg = _build_human_message(message, attachment_ctx)
        all_messages = history_messages + [human_msg]

        agent = create_agent_graph(tenant_id)
        result = await agent.ainvoke({
            "messages": all_messages,
            "tenant_id": tenant_id,
            "user_id": user_id,
            "reflection_count": 0,
            "attachment_ids": _get_attachment_ids(attachment_ctx),
        })

        final_message = result["messages"][-1]
        response_text = _get_text_content(final_message.content)
        sources = _extract_sources(result["messages"])

        assistant_msg = Message(
            conversation_id=conversation.id,
            role="assistant",
            content=response_text,
            sources=sources,
            sequence=msg_count + 1,
        )
        db.add(assistant_msg)
        await db.commit()

        return ChatResponse(
            message=response_text,
            sources=sources,
            conversation_id=str(conversation.id),
        )
    finally:
        if attachment_ctx:
            attachment_ctx.cleanup()


async def _handle_chat_stream(
    message: str,
    conversation_id: str | None,
    files: list[UploadFile],
    current_user: User,
    db: AsyncSession,
) -> StreamingResponse:
    """Core streaming chat handler."""
    tenant_id = str(current_user.tenant_id)
    user_id = str(current_user.id)

    if files:
        _validate_files(files)

    attachment_ctx = None
    if files:
        attachment_ctx = await process_attachments(files)
        
        # --- START NEW BACKGROUND INDEXING LOGIC (Option A) ---
        try:
            from app.api.documents import MIME_TYPE_MAP
            from app.models.document import Document, DocumentStatus, DocumentType
            
            storage = get_storage()
            for att in attachment_ctx.attachments:
                if att.category in ["document", "spreadsheet"]:
                    temp_path = os.path.join(attachment_ctx.temp_dir, f"{att.attachment_id}_{att.filename}")
                    with open(temp_path, "rb") as f:
                        raw_bytes = f.read()
                    
                    stored_path = storage.save(current_user.tenant_id, att.filename, raw_bytes)
                    doc_type = MIME_TYPE_MAP.get(att.mime_type, DocumentType.OTHER)
                    
                    document = Document(
                        filename=att.filename,
                        original_filename=att.filename,
                        file_path=stored_path,
                        file_size=len(raw_bytes),
                        mime_type=att.mime_type,
                        document_type=doc_type,
                        status=DocumentStatus.PENDING,
                        tenant_id=current_user.tenant_id,
                        owner_id=current_user.id,
                    )
                    db.add(document)
                    await db.flush()
                    
                    try:
                        from app.celery_app import celery_app
                        from app.worker import process_document
                        process_document.delay(str(document.id))
                        logger.info(f"[stream] Queued chat attachment {att.filename} for permanent Vector DB storage.")
                    except Exception as ce:
                        logger.error(f"[stream] Failed to queue chat attachment for Celery: {ce}")
                        
        except Exception as e:
            logger.error(f"[stream] Failed to push chat attachments to Knowledge Base pipeline: {e}")
        # --- END NEW BACKGROUND INDEXING LOGIC ---

    conversation, history_messages = await _get_or_create_conversation(
        db, current_user, conversation_id, message,
    )

    msg_count = len(history_messages)
    user_msg = Message(
        conversation_id=conversation.id,
        role="user",
        content=message,
        sequence=msg_count,
    )
    db.add(user_msg)
    await db.flush()
    await db.refresh(user_msg)

    # Persist attachments before generator so temp files still exist
    if attachment_ctx:
        records = _build_attachment_db_records(attachment_ctx, current_user.tenant_id)
        user_msg.attachments = records or None

    human_msg = _build_human_message(message, attachment_ctx)
    all_messages = history_messages + [human_msg]
    attachment_ids = _get_attachment_ids(attachment_ctx)

    async def generate():
        try:
            agent = create_agent_graph(tenant_id)
            full_response = ""
            sources = []

            async for event in agent.astream(
                {
                    "messages": all_messages,
                    "tenant_id": tenant_id,
                    "user_id": user_id,
                    "reflection_count": 0,
                    "attachment_ids": attachment_ids,
                },
                stream_mode="values",
            ):
                messages = event.get("messages", [])
                if messages:
                    last = messages[-1]
                    if hasattr(last, "content") and last.content:
                        if not hasattr(last, "tool_calls") or not last.tool_calls:
                            text = _get_text_content(last.content)
                            if text:
                                full_response = text
                                sources = _extract_sources(messages)
                                yield f"data: {json.dumps({'type': 'chunk', 'content': text})}\n\n"

            yield f"data: {json.dumps({'type': 'done', 'conversation_id': str(conversation.id), 'sources': sources})}\n\n"

            if full_response:
                async with db.begin_nested():
                    assistant_msg = Message(
                        conversation_id=conversation.id,
                        role="assistant",
                        content=full_response,
                        sources=sources,
                        sequence=msg_count + 1,
                    )
                    db.add(assistant_msg)
                await db.commit()
        finally:
            if attachment_ctx:
                attachment_ctx.cleanup()

    return StreamingResponse(generate(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Shared conversation helper
# ---------------------------------------------------------------------------
async def _get_or_create_conversation(
    db: AsyncSession,
    current_user: User,
    conversation_id: str | None,
    message: str,
) -> tuple:
    """Get existing or create new conversation. Returns (conversation, history_messages)."""
    conversation = None
    history_messages = []

    if conversation_id:
        result = await db.execute(
            select(Conversation)
            .where(Conversation.id == conversation_id)
            .where(Conversation.tenant_id == current_user.tenant_id)
            .where(Conversation.user_id == current_user.id)
        )
        conversation = result.scalar_one_or_none()
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        msg_result = await db.execute(
            select(Message)
            .where(Message.conversation_id == conversation.id)
            .order_by(Message.sequence.asc())
        )
        for msg in msg_result.scalars().all():
            if msg.role == "user":
                history_messages.append(HumanMessage(content=msg.content))
            else:
                history_messages.append(AIMessage(content=msg.content))

    if not conversation:
        conversation = Conversation(
            tenant_id=current_user.tenant_id,
            user_id=current_user.id,
            title=(message or "New conversation")[:100],
        )
        db.add(conversation)
        await db.flush()
        await db.refresh(conversation)

    return conversation, history_messages


# ---------------------------------------------------------------------------
# Conversation History Endpoints
# ---------------------------------------------------------------------------
@router.get("/conversations")
async def list_conversations(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List all conversations for the current user."""
    result = await db.execute(
        select(Conversation)
        .where(Conversation.tenant_id == current_user.tenant_id)
        .where(Conversation.user_id == current_user.id)
        .order_by(Conversation.updated_at.desc())
    )
    conversations = result.scalars().all()

    items = []
    for conv in conversations:
        msg_result = await db.execute(
            select(Message)
            .where(Message.conversation_id == conv.id)
        )
        msg_count = len(msg_result.scalars().all())

        items.append({
            "id": str(conv.id),
            "title": conv.title,
            "created_at": conv.created_at.isoformat() if conv.created_at else None,
            "updated_at": conv.updated_at.isoformat() if conv.updated_at else None,
            "message_count": msg_count,
        })

    return {"conversations": items}


@router.get("/conversations/{conversation_id}")
async def get_conversation(
    conversation_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get all messages in a conversation."""
    result = await db.execute(
        select(Conversation)
        .where(Conversation.id == conversation_id)
        .where(Conversation.tenant_id == current_user.tenant_id)
        .where(Conversation.user_id == current_user.id)
    )
    conversation = result.scalar_one_or_none()
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    msg_result = await db.execute(
        select(Message)
        .where(Message.conversation_id == conversation.id)
        .order_by(Message.sequence.asc())
    )
    messages = msg_result.scalars().all()

    return {
        "id": str(conversation.id),
        "title": conversation.title,
        "messages": [
            {
                "id": str(msg.id),
                "role": msg.role,
                "content": msg.content,
                "sources": msg.sources or [],
                "created_at": msg.created_at.isoformat() if msg.created_at else None,
                "attachments": _serialize_attachments_for_response(
                    msg.attachments, str(msg.id)
                ),
            }
            for msg in messages
        ],
    }


@router.get("/attachments")
async def serve_attachment(
    message_id: str,
    attachment_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Serve a chat attachment. Tenant-isolated via DB ownership check."""
    result = await db.execute(
        select(Message)
        .join(Conversation, Message.conversation_id == Conversation.id)
        .where(
            Message.id == uuid.UUID(message_id),
            Conversation.tenant_id == current_user.tenant_id,
            Conversation.user_id == current_user.id,
        )
    )
    msg = result.scalar_one_or_none()
    if not msg:
        raise HTTPException(status_code=404, detail="Attachment not found")

    att_meta = next(
        (a for a in (msg.attachments or []) if a["id"] == attachment_id), None
    )
    if not att_meta:
        raise HTTPException(status_code=404, detail="Attachment not found")

    if "content_b64" in att_meta:
        content = _b64.b64decode(att_meta["content_b64"])
    else:
        storage = get_storage()
        try:
            content = storage.read(current_user.tenant_id, att_meta["gcs_path"])
        except Exception:
            raise HTTPException(status_code=404, detail="File not found")

    return Response(
        content=content,
        media_type=att_meta.get("mime_type", "application/octet-stream"),
        headers={"Cache-Control": "private, max-age=3600"},
    )


@router.delete("/conversations/{conversation_id}", status_code=204)
async def delete_conversation(
    conversation_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Delete a conversation and all its messages."""
    result = await db.execute(
        select(Conversation)
        .where(Conversation.id == conversation_id)
        .where(Conversation.tenant_id == current_user.tenant_id)
        .where(Conversation.user_id == current_user.id)
    )
    conversation = result.scalar_one_or_none()
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    await db.delete(conversation)
    return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _get_text_content(content) -> str:
    """
    Normalize LLM message content to a plain string.

    Some providers (e.g. Gemini) return content as a list of blocks
    like [{'type': 'text', 'text': '...'}] instead of a plain string.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return str(content)


def _extract_sources(messages) -> list[dict]:
    """
    Extract source citations from tool call results in the message history.

    Parses the tool output messages to find [Source X: filename, Page Y] patterns.
    """
    sources = []
    seen = set()

    for msg in messages:
        if hasattr(msg, "content") and isinstance(msg.content, str):
            pattern = r'\[Source \d+: ([^,\]]+)(?:, Page (\d+))?\s*(?:\(relevance: ([\d.]+)\))?\]'
            matches = re.findall(pattern, msg.content)

            for match in matches:
                doc_name = match[0].strip()
                page = int(match[1]) if match[1] else None
                relevance = float(match[2]) if match[2] else None

                key = (doc_name, page)
                if key not in seen:
                    seen.add(key)
                    source = {"document": doc_name}
                    if page:
                        source["page"] = page
                    if relevance:
                        source["relevance"] = relevance
                    sources.append(source)

    return sources
