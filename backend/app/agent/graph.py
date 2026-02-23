"""
LangGraph ReAct Agent workflow for Automobile SSU Document Chatbot.

Graph flow:
    [Start] → decompose → agent → should_continue?
                                    ├─ tool_calls → tools → agent (loop)
                                    └─ no tool_calls → reflect → should_retry?
                                                                  ├─ incomplete → agent (max 1 retry)
                                                                  └─ adequate → [End]
"""
import logging

from google import genai
from google.genai import types
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from app.config import get_settings
from app.agent.state import AgentState
from app.agent.tools import get_agent_tools

settings = get_settings()
logger = logging.getLogger(__name__)


def _extract_text(content) -> str:
    """Extract plain text from a HumanMessage content (str or list of blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            block.get("text", "") for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return str(content) if content else ""

SYSTEM_PROMPT = """You are an expert document assistant for an automobile Shared Service Unit (SSU). You help users find and analyze information in their uploaded documents, which typically include parts catalogs, pricing tables, service manuals, specification sheets, compliance reports, and Excel/CSV data files.

## Domain Knowledge
- Part numbers follow patterns like "BRK-45821", "ENG-10034", "FLT-7890"
- Service intervals are typically measured in miles or months
- Pricing data often appears in tables with columns for part number, description, unit price, quantity
- Compliance reports reference standards like ISO, SAE, FMVSS

## Rules
1. ALWAYS use the search_documents tool to find information before answering factual questions.
2. If the user asks about a specific document, use search_specific_document with its ID.
3. Base your answers ONLY on information found in the documents. Never fabricate data.
4. If no relevant information is found, say so honestly — do not guess.
5. When presenting tabular data (prices, specs, part lists), format as a markdown table.
6. For numerical questions (totals, differences, percentages), use the calculator tool.
7. For date-related questions (warranty periods, intervals), use the date_calculator tool.
8. For "summarize this document" requests, use the summarize_document tool.
9. For comparison requests, use the compare_documents tool.
10. Always cite your sources at the end of your response:
    **Sources:**
    - [Document Name, Page X]
11. Be concise but thorough. Quote relevant passages when helpful.
12. If the user's question is conversational (greetings, thanks), respond naturally without searching.
13. When the user attaches files to their message:
    - For images: analyze them directly using your vision capability.
    - For PDFs/DOCX: the extracted text is included in the message context. Answer from it.
    - For CSV/XLSX spreadsheets: use the query_spreadsheet tool with file_identifier="attachment:<id>" for precise data queries. The attachment ID and schema are provided in the message.
14. For spreadsheet questions on permanently indexed documents, use query_spreadsheet with file_identifier="document:<document_id>".
15. Prefer query_spreadsheet over search_documents for numerical/aggregation questions about tabular data."""


async def _decompose_query(state: AgentState) -> dict:
    """
    Decompose complex multi-part queries into sub-steps.

    Simple queries pass through unchanged. Complex queries get
    decomposed into numbered sub-steps injected as a system hint.
    """
    messages = state["messages"]
    if not messages:
        return {"messages": []}

    last_msg = messages[-1]
    if not isinstance(last_msg, HumanMessage):
        return {"messages": []}

    query = _extract_text(last_msg.content)
    logger.info(f"Decompose: extracted query ({len(query)} chars): {query[:100]!r}")
    if not query or len(query) < 30:
        return {"messages": []}

    try:
        client = genai.Client(api_key=settings.google_api_key)
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=f"""Analyze this query and determine if it requires multiple distinct steps to answer.

Query: "{query}"

If the query is SIMPLE (single question, single lookup), respond with exactly: SIMPLE
If the query is COMPLEX (multiple parts, comparisons, calculations), respond with a numbered plan:
1. First step
2. Second step
...

Only respond with SIMPLE or the numbered plan. Nothing else.""",
            config=types.GenerateContentConfig(temperature=0, max_output_tokens=256),
        )

        result = response.text.strip()
        if result.upper() == "SIMPLE":
            return {"messages": []}

        # Inject decomposition as a system hint
        hint = AIMessage(content=f"I'll break this down into steps:\n{result}\n\nLet me work through each step.")
        return {"messages": [hint]}

    except Exception as e:
        logger.warning(f"Query decomposition failed: {e}")
        return {"messages": []}


def _should_continue(state: AgentState) -> str:
    """
    Decide whether to continue to tools or move to reflection.

    If the last message has tool_calls, route to the tool node.
    Otherwise, route to reflection.
    """
    last_message = state["messages"][-1]

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "reflect"


async def _call_model(state: AgentState) -> dict:
    """
    Call the LLM with the current conversation + system prompt.

    The LLM will either:
    - Generate a final response (no tool calls)
    - Request a tool call (search_documents, calculator, etc.)
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

    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    
    # Log message structure sent to LLM
    for i, msg in enumerate(messages):
        if isinstance(msg, HumanMessage):
            if isinstance(msg.content, list):
                block_types = []
                for block in msg.content:
                    if isinstance(block, dict):
                        btype = block.get("type", "unknown")
                        if btype == "image_url":
                            url = block.get("image_url", {}).get("url", "")
                            btype = f"image_url({len(url)}chars,starts={url[:30]!r})"
                        elif btype == "text":
                            btype = f"text({len(block.get('text', ''))}chars)"
                        block_types.append(btype)
                logger.info(f"Message[{i}] HumanMessage: {len(msg.content)} blocks → {block_types}")
            else:
                logger.info(f"Message[{i}] HumanMessage: plain text ({len(msg.content)}chars)")

    response = await llm_with_tools.ainvoke(messages)

    return {"messages": [response]}


async def _reflect_on_answer(state: AgentState) -> dict:
    """
    Self-reflection node: check if the answer is adequate.

    Uses a fast Gemini call to evaluate: ADEQUATE / INCOMPLETE / NO_DATA
    """
    messages = state["messages"]
    if not messages:
        return {"messages": [], "reflection_count": state.get("reflection_count", 0)}

    # Find the last AI message (the answer) and the user query
    last_ai_msg = None
    user_query = None
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and last_ai_msg is None:
            last_ai_msg = msg
        if isinstance(msg, HumanMessage) and user_query is None:
            user_query = _extract_text(msg.content)
        if last_ai_msg and user_query:
            break

    if not last_ai_msg or not user_query:
        logger.info(f"Reflect: skipping — last_ai_msg={bool(last_ai_msg)}, user_query={bool(user_query)}")
        return {"messages": [], "reflection_count": state.get("reflection_count", 0)}

    logger.info(f"Reflect: user_query ({len(user_query)} chars): {user_query[:100]!r}")

    # Skip reflection for conversational responses
    raw_content = last_ai_msg.content or ""
    answer = raw_content if isinstance(raw_content, str) else str(raw_content)
    if len(answer) < 20:
        return {"messages": [], "reflection_count": state.get("reflection_count", 0)}

    try:
        client = genai.Client(api_key=settings.google_api_key)
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=f"""Evaluate if this answer adequately addresses the user's question.

User question: "{user_query}"

Answer: "{answer[:2000]}"

Respond with exactly one word:
- ADEQUATE — if the answer fully addresses the question with data/sources
- INCOMPLETE — if the answer is partial or missing key details that could be found with more searching
- NO_DATA — if the documents genuinely don't contain the needed information

One word only:""",
            config=types.GenerateContentConfig(temperature=0, max_output_tokens=20),
        )

        if not response.text:
            logger.warning("Reflection model returned no text")
            return {"messages": [], "reflection_count": state.get("reflection_count", 0)}

        verdict = response.text.strip().upper()
        logger.info(f"Reflection verdict: {verdict}")

        current_count = state.get("reflection_count", 0)

        if (verdict.startswith("INCOMPL") or verdict.startswith("INADEQU")) and current_count < 1:
            # Send agent back for another search attempt
            retry_hint = AIMessage(
                content="Let me search more thoroughly to find additional details."
            )
            return {"messages": [retry_hint], "reflection_count": current_count + 1}

        return {"messages": [], "reflection_count": current_count}

    except Exception as e:
        logger.warning(f"Reflection failed: {e}")
        return {"messages": [], "reflection_count": state.get("reflection_count", 0)}


def _should_retry(state: AgentState) -> str:
    """
    After reflection, decide if we should retry or end.

    If reflection added a retry hint message, route back to agent.
    Otherwise, end.
    """
    last_message = state["messages"][-1]

    if isinstance(last_message, AIMessage) and "search more thoroughly" in (last_message.content or ""):
        return "agent"
    return END


def create_agent_graph(tenant_id: str) -> StateGraph:
    """
    Build the LangGraph ReAct agent for a specific tenant.

    Graph flow:
        [Start] → decompose → agent → should_continue?
                                        ├─ tool_calls → tools → agent (loop)
                                        └─ no tool_calls → reflect → should_retry?
                                                                      ├─ incomplete → agent (max 1)
                                                                      └─ adequate → [End]
    """
    tools = get_agent_tools(tenant_id)
    tool_node = ToolNode(tools)

    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("decompose", _decompose_query)
    graph.add_node("agent", _call_model)
    graph.add_node("tools", tool_node)
    graph.add_node("reflect", _reflect_on_answer)

    # Set entry point
    graph.set_entry_point("decompose")

    # Edges
    graph.add_edge("decompose", "agent")
    graph.add_conditional_edges("agent", _should_continue, {"tools": "tools", "reflect": "reflect"})
    graph.add_edge("tools", "agent")
    graph.add_conditional_edges("reflect", _should_retry, {"agent": "agent", END: END})

    return graph.compile()
