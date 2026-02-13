"""
Chat API endpoints.

Provides endpoints for chatting with the LangGraph ReAct agent,
streaming responses, and managing conversation history.
Supports both JSON and multipart/form-data (with file attachments).
"""
import json
import logging
import re

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from langchain_core.messages import HumanMessage, AIMessage

from app.config import get_settings
from app.database import get_db
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

    # Build text context from document/spreadsheet attachments
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

    full_text = (text or "").strip()
    if extra_context_parts:
        full_text += "\n" + "\n".join(extra_context_parts)

    if full_text:
        content_blocks.append({"type": "text", "text": full_text})

    # Add image data URLs
    for att in attachment_ctx.attachments:
        if att.category == "image" and att.image_data_url:
            content_blocks.append({
                "type": "image_url",
                "image_url": {"url": att.image_data_url},
            })

    if not content_blocks:
        return HumanMessage(content=text or "")

    return HumanMessage(content=content_blocks)


def _get_attachment_suffix(attachment_ctx: AttachmentContext | None) -> str:
    """Build a suffix like ' [Attached: file1.csv, photo.png]' for DB storage."""
    if not attachment_ctx or not attachment_ctx.attachments:
        return ""
    names = [a.filename for a in attachment_ctx.attachments]
    return f" [Attached: {', '.join(names)}]"


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
        files = form.getlist("files")
        # Filter to actual UploadFile objects
        files = [f for f in files if isinstance(f, UploadFile)]
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
        files = form.getlist("files")
        files = [f for f in files if isinstance(f, UploadFile)]
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
        # Process attachments
        if files:
            attachment_ctx = await process_attachments(files)

        # Get or create conversation
        conversation, history_messages = await _get_or_create_conversation(
            db, current_user, conversation_id, message,
        )

        # Save user message (text + attachment suffix for DB)
        msg_count = len(history_messages)
        db_content = message + _get_attachment_suffix(attachment_ctx)
        user_msg = Message(
            conversation_id=conversation.id,
            role="user",
            content=db_content,
            sequence=msg_count,
        )
        db.add(user_msg)

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

    conversation, history_messages = await _get_or_create_conversation(
        db, current_user, conversation_id, message,
    )

    msg_count = len(history_messages)
    db_content = message + _get_attachment_suffix(attachment_ctx)
    user_msg = Message(
        conversation_id=conversation.id,
        role="user",
        content=db_content,
        sequence=msg_count,
    )
    db.add(user_msg)
    await db.flush()

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
            }
            for msg in messages
        ],
    }


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
