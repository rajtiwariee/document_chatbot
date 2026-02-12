"""
LangGraph ReAct Agent workflow.

Implements a Reason-Act-Observe loop:
1. The LLM reasons about the user's question
2. If needed, it calls a tool (search documents)
3. It observes the tool output
4. It generates a final answer with source citations

The graph automatically loops between reasoning and acting
until the LLM decides it has enough information to answer.
"""
import logging

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from app.config import get_settings
from app.agent.state import AgentState
from app.agent.tools import get_agent_tools

settings = get_settings()
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a helpful document assistant. You help users find information in their uploaded documents.

Rules:
1. ALWAYS use the search_documents tool to find information before answering.
2. If the user asks about a specific document, use search_specific_document.
3. Base your answers ONLY on information found in the documents.
4. If no relevant information is found, say so honestly.
5. Always cite your sources at the end of your response in this format:
   **Sources:**
   - [Document Name, Page X]
6. Be concise but thorough. Quote relevant passages when helpful.
7. If the user's question is conversational (greetings, thanks), respond naturally without searching."""


def _should_continue(state: AgentState) -> str:
    """
    Decide whether to continue to tools or end.

    If the last message has tool_calls, route to the tool node.
    Otherwise, the agent is done reasoning — route to END.
    """
    last_message = state["messages"][-1]

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END


async def _call_model(state: AgentState) -> dict:
    """
    Call the LLM with the current conversation + system prompt.

    The LLM will either:
    - Generate a final response (no tool calls)
    - Request a tool call (search_documents, etc.)
    """
    tenant_id = state["tenant_id"]
    tools = get_agent_tools(tenant_id)

    llm = ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.google_api_key,
        temperature=0.3,
        convert_system_message_to_human=True,
    )

    llm_with_tools = llm.bind_tools(tools)

    # Prepend system message
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]

    response = await llm_with_tools.ainvoke(messages)

    return {"messages": [response]}


def create_agent_graph(tenant_id: str) -> StateGraph:
    """
    Build the LangGraph ReAct agent for a specific tenant.

    Graph flow:
        [Start] → agent (LLM) → should_continue?
                                    ├─ tool_calls → tools → agent (loop back)
                                    └─ no tool_calls → [End]
    """
    tools = get_agent_tools(tenant_id)
    tool_node = ToolNode(tools)

    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("agent", _call_model)
    graph.add_node("tools", tool_node)

    # Set entry point
    graph.set_entry_point("agent")

    # Add edges
    graph.add_conditional_edges("agent", _should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")  # After tools, go back to agent

    return graph.compile()
