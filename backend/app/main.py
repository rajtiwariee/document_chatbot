"""
FastAPI application entry point.
"""
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.database import init_db
from app.logging_config import setup_logging
from app.api import auth, documents, chat
from app.websockets import manager, authenticate_websocket

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    setup_logging()
    await init_db()
    yield
    # Shutdown
    pass


app = FastAPI(
    title=settings.app_name,
    description="Multi-tenant document chatbot with ReAct agent",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router, prefix="/api/auth", tags=["Authentication"])
app.include_router(documents.router, prefix="/api/documents", tags=["Documents"])
app.include_router(chat.router, prefix="/api/chat", tags=["Chat"])


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time notifications.
    Authenticate via query param: ws://host/ws?token=<jwt_token>
    """
    # Authenticate the connection
    auth_data = await authenticate_websocket(websocket)
    tenant_id = auth_data["tenant_id"]
    user_id = auth_data["user_id"]

    # Register the connection
    connected = await manager.connect(websocket, tenant_id, user_id)
    if not connected:
        return

    try:
        while True:
            # Wait for messages with heartbeat timeout (90 seconds)
            data = await asyncio.wait_for(
                websocket.receive_json(),
                timeout=manager.HEARTBEAT_TIMEOUT,
            )
            # Respond to ping with pong (heartbeat)
            if data.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
    except (asyncio.TimeoutError, Exception):
        # Timeout or disconnect — clean up
        manager.disconnect(websocket, tenant_id, user_id)


# ---------------------------------------------------------------------------
# Health endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "app": settings.app_name}


@app.get("/health/ws")
async def websocket_stats():
    """WebSocket connection statistics."""
    return manager.get_stats()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)