"""
Chat API endpoints.
"""
from pydantic import BaseModel
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.models.user import User
from app.api.auth import get_current_user

router = APIRouter()


class ChatRequest(BaseModel):
    """Chat request schema."""
    message: str
    conversation_id: str | None = None


class ChatResponse(BaseModel):
    """Chat response schema."""
    message: str
    sources: list[dict] = []
    conversation_id: str


@router.post("/", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    current_user: User = Depends(get_current_user)
):
    """Send a message to the chatbot."""
    # TODO: Implement LangGraph ReAct agent
    return ChatResponse(
        message="Chat endpoint - LangGraph agent implementation pending",
        sources=[],
        conversation_id=request.conversation_id or "new-conversation"
    )


@router.post("/stream")
async def chat_stream(
    request: ChatRequest,
    current_user: User = Depends(get_current_user)
):
    """Stream chat response."""
    # TODO: Implement streaming with LangGraph
    async def generate():
        yield "data: Chat streaming - implementation pending\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")
