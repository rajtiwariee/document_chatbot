"""
Agent state schema for LangGraph.

Defines the state that flows through the ReAct agent graph.
"""
from typing import Annotated
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """
    State passed between nodes in the LangGraph agent.

    - messages: Full conversation history (LangChain message format).
                Uses `add_messages` reducer to append new messages.
    - tenant_id: Tenant scope for vector search isolation.
    - user_id: User who initiated the chat.
    - reflection_count: Number of self-reflection retries (max 1).
    """
    messages: Annotated[list[BaseMessage], add_messages]
    tenant_id: str
    user_id: str
    reflection_count: int
