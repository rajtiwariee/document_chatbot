"""
WebSocket connection manager for real-time notifications.

Provides tenant-isolated WebSocket connections for:
- Document processing progress updates
- Task completion notifications
- Error alerts

Authentication: JWT token passed as query parameter (?token=xxx)
"""
import asyncio
import json
import logging
from typing import Dict, Set

from fastapi import WebSocket, WebSocketException, status
from jose import jwt, JWTError

from app.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)


class ConnectionManager:
    """
    Manages WebSocket connections for real-time notifications.
    Maintains a mapping of tenant_id -> set of active websockets.
    """

    # Limits
    MAX_CONNECTIONS_PER_TENANT = 500
    MAX_CONNECTIONS_PER_USER = 3
    HEARTBEAT_TIMEOUT = 90  # seconds

    def __init__(self):
        # tenant_id -> set of (websocket, user_id) tuples
        self.active_connections: Dict[str, Set[tuple]] = {}
        # user_id -> count of active connections
        self.user_connection_count: Dict[str, int] = {}

    def _tenant_count(self, tenant_id: str) -> int:
        return len(self.active_connections.get(tenant_id, set()))

    def _user_count(self, user_id: str) -> int:
        return self.user_connection_count.get(user_id, 0)

    async def connect(
        self, websocket: WebSocket, tenant_id: str, user_id: str
    ) -> bool:
        """
        Accept and register a WebSocket connection.
        Returns False if connection limits are exceeded.
        """
        # Check tenant limit
        if self._tenant_count(tenant_id) >= self.MAX_CONNECTIONS_PER_TENANT:
            await websocket.close(code=1013, reason="Too many tenant connections")
            return False

        # Check per-user limit
        if self._user_count(user_id) >= self.MAX_CONNECTIONS_PER_USER:
            await websocket.close(code=1013, reason="Too many connections for user")
            return False

        await websocket.accept()

        if tenant_id not in self.active_connections:
            self.active_connections[tenant_id] = set()
        self.active_connections[tenant_id].add((websocket, user_id))

        self.user_connection_count[user_id] = self._user_count(user_id) + 1

        logger.info(
            f"WebSocket connected: tenant={tenant_id}, user={user_id}, "
            f"tenant_total={self._tenant_count(tenant_id)}"
        )
        return True

    def disconnect(self, websocket: WebSocket, tenant_id: str, user_id: str):
        """Remove a WebSocket connection."""
        if tenant_id in self.active_connections:
            self.active_connections[tenant_id].discard((websocket, user_id))
            if not self.active_connections[tenant_id]:
                del self.active_connections[tenant_id]

        if user_id in self.user_connection_count:
            self.user_connection_count[user_id] = max(
                0, self.user_connection_count[user_id] - 1
            )
            if self.user_connection_count[user_id] == 0:
                del self.user_connection_count[user_id]

        logger.info(f"WebSocket disconnected: tenant={tenant_id}, user={user_id}")

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        """Send a JSON message to a specific connection."""
        try:
            await websocket.send_json(message)
        except Exception:
            pass

    async def broadcast_to_tenant(self, message: dict, tenant_id: str):
        """Broadcast a JSON message to all users in a tenant."""
        if tenant_id not in self.active_connections:
            return

        dead_connections = set()
        for ws, user_id in self.active_connections[tenant_id]:
            try:
                await ws.send_json(message)
            except Exception:
                dead_connections.add((ws, user_id))

        # Clean up dead connections
        for ws, user_id in dead_connections:
            self.disconnect(ws, tenant_id, user_id)

    async def send_to_user(self, message: dict, tenant_id: str, target_user_id: str):
        """Send a message to all connections of a specific user within a tenant."""
        if tenant_id not in self.active_connections:
            return

        for ws, user_id in self.active_connections[tenant_id]:
            if user_id == target_user_id:
                try:
                    await ws.send_json(message)
                except Exception:
                    pass

    def get_stats(self) -> dict:
        """Get connection statistics for monitoring."""
        return {
            "total_tenants": len(self.active_connections),
            "total_connections": sum(
                len(conns) for conns in self.active_connections.values()
            ),
            "total_users": len(self.user_connection_count),
        }


# Global manager instance
manager = ConnectionManager()


# ---------------------------------------------------------------------------
# WebSocket authentication
# ---------------------------------------------------------------------------
async def authenticate_websocket(websocket: WebSocket) -> dict:
    """
    Authenticate a WebSocket connection using JWT from query parameter.
    Usage: ws://host/ws?token=<jwt_token>

    Returns dict with user_id and tenant_id.
    """
    token = websocket.query_params.get("token")
    if not token:
        raise WebSocketException(code=status.WS_1008_POLICY_VIOLATION)

    try:
        payload = jwt.decode(
            token, settings.secret_key, algorithms=[settings.algorithm]
        )
        user_id = payload.get("sub")
        tenant_id = payload.get("tenant_id")

        if not user_id or not tenant_id:
            raise WebSocketException(code=status.WS_1008_POLICY_VIOLATION)

        return {"user_id": user_id, "tenant_id": tenant_id}
    except JWTError:
        raise WebSocketException(code=status.WS_1008_POLICY_VIOLATION)


# ---------------------------------------------------------------------------
# Notification helpers (called from Celery workers or API endpoints)
# ---------------------------------------------------------------------------
async def notify_document_progress(
    tenant_id: str,
    user_id: str,
    document_id: str,
    stage: str,
    percent: int,
    message: str = "",
):
    """
    Send document processing progress to the uploading user.

    Stages: uploading, extracting, chunking, embedding, indexing, completed, failed
    """
    await manager.send_to_user(
        message={
            "type": "document_progress",
            "document_id": document_id,
            "stage": stage,
            "percent": percent,
            "message": message,
        },
        tenant_id=tenant_id,
        target_user_id=user_id,
    )


async def notify_document_completed(
    tenant_id: str, document_id: str, filename: str
):
    """Broadcast to all tenant users that a new document is ready."""
    await manager.broadcast_to_tenant(
        message={
            "type": "document_completed",
            "document_id": document_id,
            "filename": filename,
            "message": f"Document '{filename}' is now ready for search.",
        },
        tenant_id=tenant_id,
    )


async def notify_document_failed(
    tenant_id: str, user_id: str, document_id: str, error: str
):
    """Notify the uploading user that document processing failed."""
    await manager.send_to_user(
        message={
            "type": "document_failed",
            "document_id": document_id,
            "error": error,
            "message": f"Document processing failed: {error}",
        },
        tenant_id=tenant_id,
        target_user_id=user_id,
    )
