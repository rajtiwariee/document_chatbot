"""
Chat API endpoints.

Provides endpoints for chatting with the LangGraph ReAct agent,
streaming responses, and managing conversation history.
"""
import json
import uuid
import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from langchain_core.messages import HumanMessage, AIMessage

from app.database import get_db
from app.models.user import User
from app.models.conversation import Conversation, Message
from app.api.auth import get_current_user
from app.agent.graph import create_agent_graph

logger = logging.getLogger(__name__)

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
# Chat Endpoints
# ---------------------------------------------------------------------------
@router.post("/", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Send a message to the document assistant.
    Creates a new conversation or continues an existing one.
    """
    tenant_id = str(current_user.tenant_id)
    user_id = str(current_user.id)

    # Get or create conversation
    conversation = None
    history_messages = []

    if request.conversation_id:
        result = await db.execute(
            select(Conversation)
            .where(Conversation.id == request.conversation_id)
            .where(Conversation.tenant_id == current_user.tenant_id)
            .where(Conversation.user_id == current_user.id)
        )
        conversation = result.scalar_one_or_none()
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        # Load conversation history
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
        # Create new conversation
        conversation = Conversation(
            tenant_id=current_user.tenant_id,
            user_id=current_user.id,
            title=request.message[:100],
        )
        db.add(conversation)
        await db.flush()
        await db.refresh(conversation)

    # Save user message
    msg_count = len(history_messages)
    user_msg = Message(
        conversation_id=conversation.id,
        role="user",
        content=request.message,
        sequence=msg_count,
    )
    db.add(user_msg)

    # Build agent state with history + new message
    all_messages = history_messages + [HumanMessage(content=request.message)]

    # Run the agent
    agent = create_agent_graph(tenant_id)
    result = await agent.ainvoke({
        "messages": all_messages,
        "tenant_id": tenant_id,
        "user_id": user_id,
        "reflection_count": 0,
    })

    # Extract the final response
    final_message = result["messages"][-1]
    response_text = _get_text_content(final_message.content)

    # Extract sources from tool calls in the message history
    sources = _extract_sources(result["messages"])

    # Save assistant message
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


@router.post("/stream")
async def chat_stream(
    request: ChatRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Stream chat response via Server-Sent Events (SSE).
    The frontend receives chunks as they are generated.
    """
    tenant_id = str(current_user.tenant_id)
    user_id = str(current_user.id)

    # Get or create conversation (same logic as non-streaming)
    conversation = None
    history_messages = []

    if request.conversation_id:
        result = await db.execute(
            select(Conversation)
            .where(Conversation.id == request.conversation_id)
            .where(Conversation.tenant_id == current_user.tenant_id)
            .where(Conversation.user_id == current_user.id)
        )
        conversation = result.scalar_one_or_none()
        if conversation:
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
            title=request.message[:100],
        )
        db.add(conversation)
        await db.flush()
        await db.refresh(conversation)

    # Save user message
    msg_count = len(history_messages)
    user_msg = Message(
        conversation_id=conversation.id,
        role="user",
        content=request.message,
        sequence=msg_count,
    )
    db.add(user_msg)
    await db.flush()

    all_messages = history_messages + [HumanMessage(content=request.message)]

    async def generate():
        agent = create_agent_graph(tenant_id)
        full_response = ""
        sources = []

        # Stream the agent execution
        async for event in agent.astream(
            {
                "messages": all_messages,
                "tenant_id": tenant_id,
                "user_id": user_id,
                "reflection_count": 0,
            },
            stream_mode="values",
        ):
            messages = event.get("messages", [])
            if messages:
                last = messages[-1]
                if hasattr(last, "content") and last.content:
                    if not hasattr(last, "tool_calls") or not last.tool_calls:
                        # This is a final text response
                        text = _get_text_content(last.content)
                        if text:
                            full_response = text
                            sources = _extract_sources(messages)

                            yield f"data: {json.dumps({'type': 'chunk', 'content': text})}\n\n"

        # Send final event with sources and conversation_id
        yield f"data: {json.dumps({'type': 'done', 'conversation_id': str(conversation.id), 'sources': sources})}\n\n"

        # Save assistant message (fire-and-forget within the generator)
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

    return StreamingResponse(generate(), media_type="text/event-stream")


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
            # Look for [Source N: filename, Page X] patterns
            import re
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
