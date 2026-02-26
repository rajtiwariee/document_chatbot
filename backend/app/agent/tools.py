"""
Agent tools for document retrieval, computation, and analysis.

These tools are used by the LangGraph ReAct agent to search,
retrieve, compute, and analyze information from the tenant's
indexed documents.
"""
import ast
import io
import logging
import math
import operator
import re
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
from langchain_core.tools import tool
from sqlalchemy import select

from app.config import get_settings
from app.ai_client import get_genai_client
from app.document_processing.embeddings import EmbeddingGenerator
from app.vector_store.store import TenantVectorStore
from app.vector_store.hybrid_search import HybridSearcher
from app.vector_store.reranker import GeminiReranker
from app.models.document import Document
from app.database import async_session_maker
from app.agent.attachment_processor import get_dataframe
from app.agent.spreadsheet_tool import execute_query

settings = get_settings()
logger = logging.getLogger(__name__)

# Shared instances (created once, reused across tool calls)
_embedding_generator = None
_vector_store = None
_hybrid_searcher = None
_reranker = None


def _get_embedding_generator() -> EmbeddingGenerator:
    global _embedding_generator
    if _embedding_generator is None:
        _embedding_generator = EmbeddingGenerator()
    return _embedding_generator


def _get_vector_store() -> TenantVectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = TenantVectorStore()
    return _vector_store


def _get_hybrid_searcher() -> HybridSearcher:
    global _hybrid_searcher
    if _hybrid_searcher is None:
        _hybrid_searcher = HybridSearcher(
            vector_store=_get_vector_store(),
            embedding_generator=_get_embedding_generator(),
        )
    return _hybrid_searcher


def _get_reranker() -> GeminiReranker:
    global _reranker
    if _reranker is None:
        _reranker = GeminiReranker(top_n=5)
    return _reranker


def _should_rerank(results: list) -> bool:
    """Only rerank when results are ambiguous (close scores)."""
    if len(results) <= 3:
        return False
    # If top result score is much higher than 3rd, ranking is already clear
    if results[0].score > results[2].score * 1.5:
        return False
    return True


def _format_search_results(results, include_score: bool = True) -> str:
    """Format search results with citations."""
    if not results:
        return "No relevant documents found for this query."

    formatted = []
    for i, result in enumerate(results, 1):
        source = result.metadata.get("source_file", "Unknown")
        file_type = result.metadata.get("file_type", "")
        sheet_name = result.metadata.get("sheet_name", "")

        # For Excel/CSV: show "Sheet: X" instead of "Page N"
        if sheet_name:
            location = f", Sheet: {sheet_name}"
        elif result.page_number:
            location = f", Page {result.page_number}"
        else:
            location = ""

        section = result.metadata.get("section_header", "")
        # Don't repeat section if it's the same as sheet name
        if section and section != f"Sheet: {sheet_name}":
            section_str = f", Section: {section}"
        else:
            section_str = ""

        el_type = result.metadata.get("element_type", "text")
        type_str = f" [{el_type}]" if el_type != "text" else ""

        header = f"[Source {i}: {source}{location}{section_str}{type_str}]"
        if include_score:
            header = f"[Source {i}: {source}{location}{section_str}{type_str} (relevance: {result.score:.2f})]"

        formatted.append(f"{header}\n{result.chunk_text}\n")

    return "\n---\n".join(formatted)


def _enrich_image_results(results: list, query: str) -> list:
    """
    For search results that came from image captions, run VQA against
    the original image to provide richer context to the LLM.
    """
    import asyncio
    import os

    enriched_any = False
    for result in results:
        content_type = result.metadata.get("content_type", "text")
        image_path = result.metadata.get("original_image_path", "")

        if content_type == "image_caption" and image_path and os.path.exists(image_path):
            try:
                from app.vision.base import get_vision_backend

                vision = get_vision_backend()
                loop = asyncio.new_event_loop()
                try:
                    vqa_answer = loop.run_until_complete(
                        vision.visual_qa(image_path, query)
                    )
                finally:
                    loop.close()

                result.chunk_text += f"\n\n[Visual Q&A — from original image]: {vqa_answer}"
                enriched_any = True
            except Exception as e:
                logger.warning("VQA enrichment failed for %s: %s", image_path, e)

    if enriched_any:
        logger.info("Enriched %d image results with VQA", sum(
            1 for r in results if r.metadata.get("content_type") == "image_caption"
        ))

    return results


def create_search_tool(tenant_id: str):
    """Create a document search tool bound to a specific tenant."""

    @tool
    def search_documents(query: str) -> str:
        """
        Search across all indexed documents for relevant information.

        Use this tool when you need to find specific facts, data,
        or context from the user's uploaded documents. The search
        uses hybrid retrieval (semantic + keyword) for best results.

        Args:
            query: A natural language search query describing what
                   information you are looking for.

        Returns:
            Relevant document passages with source citations.
        """
        logger.info(f"Tool search_documents called: query='{query}', tenant={tenant_id}")

        searcher = _get_hybrid_searcher()
        results = searcher.search(
            tenant_id=tenant_id,
            query=query,
            top_k=10,
        )

        # Rerank only when results are ambiguous
        if settings.enable_reranking and results and _should_rerank(results):
            reranker = _get_reranker()
            results = reranker.rerank(query, results)
        else:
            results = results[:5]

        # Enrich image caption results with VQA from original image
        results = _enrich_image_results(results, query)

        return _format_search_results(results)

    return search_documents


def create_document_search_tool(tenant_id: str):
    """Create a tool to search within a specific document."""

    @tool
    def search_specific_document(query: str, document_id: str) -> str:
        """
        Search within a single specific document for relevant information.

        Use this when the user asks about a particular document by name
        or when you need to dig deeper into a specific source.

        Args:
            query: What to search for within the document.
            document_id: The UUID of the document to search within.

        Returns:
            Relevant passages from that specific document.
        """
        logger.info(
            f"Tool search_specific_document: query='{query}', "
            f"doc={document_id}, tenant={tenant_id}"
        )

        searcher = _get_hybrid_searcher()
        results = searcher.search(
            tenant_id=tenant_id,
            query=query,
            top_k=10,
            document_id=document_id,
        )

        if settings.enable_reranking and results and _should_rerank(results):
            reranker = _get_reranker()
            results = reranker.rerank(query, results)
        else:
            results = results[:5]

        if not results:
            return f"No relevant content found in document {document_id}."

        formatted = []
        for i, result in enumerate(results, 1):
            sheet = result.metadata.get("sheet_name", "")
            if sheet:
                loc = f"Sheet: {sheet}"
            elif result.page_number:
                loc = f"Page {result.page_number}"
            else:
                loc = "N/A"
            el_type = result.metadata.get("element_type", "text")
            type_str = f" [{el_type}]" if el_type != "text" else ""
            formatted.append(
                f"[Passage {i}, {loc}{type_str}]\n{result.chunk_text}\n"
            )

        return "\n---\n".join(formatted)

    return search_specific_document


def create_list_documents_tool(tenant_id: str):
    """Create a tool to list all available documents for the tenant."""

    @tool
    async def list_documents() -> str:
        """
        List all documents uploaded by the user.

        Use this tool when the user asks "What documents do I have?"
        or "List my files" to see what information is available.
        """
        logger.info(f"Tool list_documents called for tenant={tenant_id}")

        async with async_session_maker() as session:
            try:
                stmt = select(Document).where(Document.tenant_id == tenant_id)
                result = await session.execute(stmt)
                docs = result.scalars().all()

                if not docs:
                    return "No documents found."

                formatted = []
                for i, doc in enumerate(docs, 1):
                    created = doc.created_at.strftime("%Y-%m-%d") if doc.created_at else "?"
                    doc_type = doc.document_type.value if doc.document_type else "unknown"
                    formatted.append(
                        f"{i}. {doc.original_filename} ({doc_type}, {created})\n"
                        f"   ID: {doc.id}"
                    )

                return "Here are your available documents:\n" + "\n".join(formatted)
            except Exception as e:
                logger.error(f"Error listing documents: {e}")
                return f"Error retrieving document list: {e}"

    return list_documents


def create_calculator_tool():
    """Create a safe math calculator tool."""

    # Allowed operations for safe evaluation
    SAFE_OPS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
        ast.Mod: operator.mod,
        ast.FloorDiv: operator.floordiv,
    }

    SAFE_FUNCTIONS = {
        "sqrt": math.sqrt,
        "round": round,
        "abs": abs,
        "min": min,
        "max": max,
        "pow": pow,
        "ceil": math.ceil,
        "floor": math.floor,
        "log": math.log,
        "log10": math.log10,
    }

    SAFE_CONSTANTS = {
        "pi": math.pi,
        "e": math.e,
    }

    def _safe_eval(node):
        """Recursively evaluate an AST node safely."""
        if isinstance(node, ast.Expression):
            return _safe_eval(node.body)
        elif isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError(f"Unsupported constant: {node.value}")
        elif isinstance(node, ast.Name):
            if node.id in SAFE_CONSTANTS:
                return SAFE_CONSTANTS[node.id]
            raise ValueError(f"Unknown variable: {node.id}")
        elif isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in SAFE_OPS:
                raise ValueError(f"Unsupported operation: {op_type.__name__}")
            left = _safe_eval(node.left)
            right = _safe_eval(node.right)
            return SAFE_OPS[op_type](left, right)
        elif isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in SAFE_OPS:
                raise ValueError(f"Unsupported operation: {op_type.__name__}")
            operand = _safe_eval(node.operand)
            return SAFE_OPS[op_type](operand)
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in SAFE_FUNCTIONS:
                args = [_safe_eval(arg) for arg in node.args]
                return SAFE_FUNCTIONS[node.func.id](*args)
            raise ValueError(f"Unsupported function call")
        else:
            raise ValueError(f"Unsupported expression type: {type(node).__name__}")

    @tool
    def calculator(expression: str) -> str:
        """
        Evaluate a mathematical expression safely.

        Supports: +, -, *, /, **, %, //
        Functions: sqrt, round, abs, min, max, pow, ceil, floor, log, log10
        Constants: pi, e

        Use this tool for computing totals, price differences, percentages,
        unit conversions, or any arithmetic from document data.

        Args:
            expression: A mathematical expression like "1500 * 0.18" or "sqrt(144) + 5"

        Returns:
            The computed result as a string.
        """
        logger.info(f"Tool calculator called: expression='{expression}'")

        try:
            tree = ast.parse(expression, mode="eval")
            result = _safe_eval(tree)

            # Format nicely
            if isinstance(result, float) and result == int(result):
                return str(int(result))
            if isinstance(result, float):
                return f"{result:.6g}"
            return str(result)
        except Exception as e:
            return f"Error evaluating expression: {e}"

    return calculator


def create_date_calculator_tool():
    """Create a date calculation tool."""

    DATE_FORMATS = [
        "%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y",
        "%B %d, %Y", "%b %d, %Y",
        "%Y-%m-%dT%H:%M:%S",
    ]

    def _parse_date(date_str: str) -> datetime:
        date_str = date_str.strip()
        for fmt in DATE_FORMATS:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        raise ValueError(f"Could not parse date: '{date_str}'")

    @tool
    def date_calculator(operation: str) -> str:
        """
        Perform date calculations.

        Supported operations:
        - "days between DATE1 and DATE2" — compute days between two dates
        - "add N days to DATE" — add days to a date
        - "subtract N days from DATE" — subtract days from a date
        - "today" — return today's date

        Date formats supported: YYYY-MM-DD, MM/DD/YYYY, DD/MM/YYYY, "Month Day, Year"

        Use this for warranty period calculations, service intervals, or deadline tracking.

        Args:
            operation: A natural language date operation.

        Returns:
            The result of the date calculation.
        """
        logger.info(f"Tool date_calculator called: operation='{operation}'")

        op = operation.strip().lower()

        if op == "today":
            return datetime.now().strftime("%Y-%m-%d")

        # "days between DATE1 and DATE2"
        m = re.match(r"days?\s+between\s+(.+?)\s+and\s+(.+)", op, re.IGNORECASE)
        if m:
            try:
                d1 = _parse_date(m.group(1))
                d2 = _parse_date(m.group(2))
                diff = abs((d2 - d1).days)
                return f"{diff} days between {d1.strftime('%Y-%m-%d')} and {d2.strftime('%Y-%m-%d')}"
            except ValueError as e:
                return str(e)

        # "add N days to DATE"
        m = re.match(r"add\s+(\d+)\s+days?\s+to\s+(.+)", op, re.IGNORECASE)
        if m:
            try:
                n = int(m.group(1))
                d = _parse_date(m.group(2))
                result = d + timedelta(days=n)
                return result.strftime("%Y-%m-%d")
            except ValueError as e:
                return str(e)

        # "subtract N days from DATE"
        m = re.match(r"subtract\s+(\d+)\s+days?\s+from\s+(.+)", op, re.IGNORECASE)
        if m:
            try:
                n = int(m.group(1))
                d = _parse_date(m.group(2))
                result = d - timedelta(days=n)
                return result.strftime("%Y-%m-%d")
            except ValueError as e:
                return str(e)

        return f"Unrecognized date operation: '{operation}'. Use 'days between DATE and DATE', 'add N days to DATE', or 'subtract N days from DATE'."

    return date_calculator


def create_summarize_document_tool(tenant_id: str):
    """Create a tool to summarize an entire document."""

    @tool
    def summarize_document(document_id: str) -> str:
        """
        Summarize an entire document by retrieving all its chunks.

        Use this when the user asks "What is this document about?" or
        "Summarize the Q2 report" or any high-level overview request.

        Args:
            document_id: The UUID of the document to summarize.

        Returns:
            A concise summary of the document's contents.
        """
        logger.info(f"Tool summarize_document called: doc={document_id}, tenant={tenant_id}")

        from google.genai import types

        store = _get_vector_store()
        collection_name = store._collection_name(tenant_id)

        try:
            store.client.get_collection(collection_name)
        except Exception:
            return f"No indexed content found for document {document_id}."

        # Scroll all chunks for this document
        all_texts = []
        offset = None
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        doc_filter = Filter(
            must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]
        )

        while True:
            results, next_offset = store.client.scroll(
                collection_name=collection_name,
                scroll_filter=doc_filter,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            if not results:
                break

            for point in results:
                text = point.payload.get("text", "")
                if text:
                    all_texts.append(text)

            offset = next_offset
            if offset is None:
                break

        if not all_texts:
            return f"No content found for document {document_id}."

        # Concatenate and truncate to ~30k chars
        combined = "\n\n---\n\n".join(all_texts)
        if len(combined) > 30000:
            combined = combined[:30000] + "\n\n[Content truncated...]"

        client = get_genai_client()
        response = client.models.generate_content(
            model=settings.gemini_model,
            contents=f"Provide a comprehensive summary of the following document content. "
            f"Highlight key topics, data points, and important findings.\n\n{combined}",
            config=types.GenerateContentConfig(temperature=0.3, max_output_tokens=1024),
        )

        return response.text

    return summarize_document


def create_compare_documents_tool(tenant_id: str):
    """Create a tool to compare two documents."""

    @tool
    def compare_documents(document_id_1: str, document_id_2: str, aspect: str = "") -> str:
        """
        Compare two documents to identify similarities and differences.

        Use this when the user asks to compare pricing between catalogs,
        differences between spec sheets, or changes between document versions.

        Args:
            document_id_1: UUID of the first document.
            document_id_2: UUID of the second document.
            aspect: Optional aspect to focus comparison on (e.g., "pricing", "specifications").

        Returns:
            A comparison of the two documents.
        """
        logger.info(
            f"Tool compare_documents: doc1={document_id_1}, doc2={document_id_2}, "
            f"aspect='{aspect}', tenant={tenant_id}"
        )

        from google.genai import types
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        store = _get_vector_store()
        collection_name = store._collection_name(tenant_id)

        def _get_doc_content(doc_id: str, max_chars: int = 15000) -> str:
            texts = []
            offset = None
            doc_filter = Filter(
                must=[FieldCondition(key="document_id", match=MatchValue(value=doc_id))]
            )
            while True:
                results, next_offset = store.client.scroll(
                    collection_name=collection_name,
                    scroll_filter=doc_filter,
                    limit=100,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
                if not results:
                    break
                for point in results:
                    text = point.payload.get("text", "")
                    if text:
                        texts.append(text)
                offset = next_offset
                if offset is None:
                    break

            combined = "\n\n".join(texts)
            return combined[:max_chars] if len(combined) > max_chars else combined

        content1 = _get_doc_content(document_id_1)
        content2 = _get_doc_content(document_id_2)

        if not content1:
            return f"No content found for document {document_id_1}."
        if not content2:
            return f"No content found for document {document_id_2}."

        aspect_str = f" Focus specifically on: {aspect}." if aspect else ""

        client = get_genai_client()
        response = client.models.generate_content(
            model=settings.gemini_model,
            contents=f"Compare the following two documents. Identify key similarities and differences.{aspect_str}\n\n"
            f"=== DOCUMENT 1 ===\n{content1}\n\n"
            f"=== DOCUMENT 2 ===\n{content2}",
            config=types.GenerateContentConfig(temperature=0.3, max_output_tokens=1500),
        )

        return response.text

    return compare_documents


def create_spreadsheet_query_tool(tenant_id: str):
    """Create a tool to query spreadsheet data using natural language."""

    @tool
    async def query_spreadsheet(question: str, file_identifier: str) -> str:
        """
        Query a CSV or XLSX spreadsheet using natural language.

        Use this tool for precise data queries on tabular data: totals,
        averages, filtering, grouping, sorting, counting, etc.
        Prefer this over search_documents for numerical/aggregation questions.

        Args:
            question: Natural language question about the data, e.g.
                      "What is the total price?" or "How many rows have status=Active?"
            file_identifier: Either "attachment:<uuid>" for a chat-attached file,
                            or "document:<uuid>" for a permanently indexed document.

        Returns:
            The query result with the generated pandas code.
        """
        logger.info(
            f"Tool query_spreadsheet: question='{question}', "
            f"file={file_identifier}, tenant={tenant_id}"
        )

        df = None

        if file_identifier.startswith("attachment:"):
            att_id = file_identifier.split(":", 1)[1]
            df = get_dataframe(att_id)
            if df is None:
                return f"Attachment {att_id} not found or not a spreadsheet."

        elif file_identifier.startswith("document:"):
            doc_id = file_identifier.split(":", 1)[1]
            try:
                async with async_session_maker() as session:
                    stmt = (
                        select(Document)
                        .where(Document.id == doc_id)
                        .where(Document.tenant_id == tenant_id)
                    )
                    result = await session.execute(stmt)
                    doc = result.scalar_one_or_none()

                    if not doc:
                        return f"Document {doc_id} not found."

                    ext = doc.original_filename.rsplit(".", 1)[-1].lower()
                    if ext not in ("csv", "xlsx"):
                        return f"Document {doc.original_filename} is not a spreadsheet (type: {ext})."

                    from app.middleware.file_storage import get_storage
                    storage = get_storage()
                    file_bytes = storage.read(doc.tenant_id, doc.file_path)

                    if ext == "csv":
                        df = pd.read_csv(io.BytesIO(file_bytes))
                    else:
                        df = pd.read_excel(io.BytesIO(file_bytes))

            except Exception as e:
                logger.error(f"Error loading document for spreadsheet query: {e}")
                return f"Error loading document: {e}"
        else:
            return f"Invalid file_identifier format: {file_identifier}. Use 'attachment:<id>' or 'document:<id>'."

        return execute_query(df, question)

    return query_spreadsheet


def get_agent_tools(tenant_id: str) -> list:
    """Get all tools for the agent, bound to a tenant."""
    return [
        create_list_documents_tool(tenant_id),
        create_search_tool(tenant_id),
        create_document_search_tool(tenant_id),
        create_calculator_tool(),
        create_date_calculator_tool(),
        create_summarize_document_tool(tenant_id),
        create_compare_documents_tool(tenant_id),
        create_spreadsheet_query_tool(tenant_id),
    ]
