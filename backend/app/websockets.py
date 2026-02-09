from typing import Dict, List
from fastapi import WebSocket

class ConnectionManager:
    """
    Manages WebSocket connections for real-time notifications.
    Maintains a mapping of tenant_id -> list of active websockets.
    """
    def __init__(self):
        # Map tenant_id to list of active connections
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, tenant_id: str):
        """Accept connection and add to tenant's list."""
        await websocket.accept()
        if tenant_id not in self.active_connections:
            self.active_connections[tenant_id] = []
        self.active_connections[tenant_id].append(websocket)

    def disconnect(self, websocket: WebSocket, tenant_id: str):
        """Remove connection from tenant's list."""
        if tenant_id in self.active_connections:
            if websocket in self.active_connections[tenant_id]:
                self.active_connections[tenant_id].remove(websocket)
            if not self.active_connections[tenant_id]:
                del self.active_connections[tenant_id]

    async def send_personal_message(self, message: str, websocket: WebSocket):
        """Send a message to a specific connection."""
        await websocket.send_text(message)

    async def broadcast_to_tenant(self, message: dict, tenant_id: str):
        """Broadcast a message (JSON) to all users in a tenant."""
        if tenant_id in self.active_connections:
            for connection in self.active_connections[tenant_id]:
                try:
                    await connection.send_json(message)
                except Exception:
                    # Handle disconnected clients gracefully
                    pass

manager = ConnectionManager()
