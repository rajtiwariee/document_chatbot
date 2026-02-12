"""
Agent tools for document retrieval.

These tools are used by the LangGraph ReAct agent to search
and retrieve information from the tenant's indexed documents.
"""
import logging
from typing import Optional

from langchain_core.tools import tool
from sqlalchemy import select

from app.config import get_settings
from app.document_processing.embeddings import EmbeddingGenerator
from app.vector_store.store import TenantVectorStore
from app.models.document import Document
from app.database import async_session_maker

settings = get_settings()
logger = logging.getLogger(__name__)

# Shared instances (created once, reused across tool calls)
_embedding_generator = None
_vector_store = None


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


def create_search_tool(tenant_id: str):
    """
    Create a document search tool bound to a specific tenant.
    """

    @tool
    def search_documents(query: str) -> str:
        """
        Search across all indexed documents for relevant information.

        Use this tool when you need to find specific facts, data,
        or context from the user's uploaded documents. The search
        uses semantic similarity to find the most relevant passages.

        Args:
            query: A natural language search query describing what
                   information you are looking for.

        Returns:
            Relevant document passages with source citations.
        """
        logger.info(f"Tool search_documents called: query='{query}', tenant={tenant_id}")

        generator = _get_embedding_generator()
        store = _get_vector_store()

        # Embed the query
        query_embedding = generator.embed_query(query)

        # Search the tenant's vector store
        results = store.search(
            tenant_id=tenant_id,
            query_embedding=query_embedding,
            top_k=5,
        )

        if not results:
            return "No relevant documents found for this query."

        # Format results with citations
        formatted = []
        for i, result in enumerate(results, 1):
            source = result.metadata.get("source_file", "Unknown")
            page = f", Page {result.page_number}" if result.page_number else ""
            score = f"{result.score:.2f}"

            formatted.append(
                f"[Source {i}: {source}{page} (relevance: {score})]\n"
                f"{result.chunk_text}\n"
            )

        return "\n---\n".join(formatted)

    return search_documents


def create_document_search_tool(tenant_id: str):
    """
    Create a tool to search within a specific document.
    """

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

        generator = _get_embedding_generator()
        store = _get_vector_store()

        query_embedding = generator.embed_query(query)

        results = store.search(
            tenant_id=tenant_id,
            query_embedding=query_embedding,
            top_k=5,
            document_id=document_id,
        )

        if not results:
            return f"No relevant content found in document {document_id}."

        formatted = []
        for i, result in enumerate(results, 1):
            page = f"Page {result.page_number}" if result.page_number else "N/A"
            formatted.append(
                f"[Passage {i}, {page}]\n{result.chunk_text}\n"
            )

        return "\n---\n".join(formatted)

    return search_specific_document


def create_list_documents_tool(tenant_id: str):
    """
    Create a tool to list all available documents for the tenant.
    """

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
                    # Format: 1. Report.pdf (PDF, 2023-10-27) - ID: ...
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
            # No finally block needed, async context manager handles close

    return list_documents


def get_agent_tools(tenant_id: str) -> list:
    """Get all tools for the agent, bound to a tenant."""
    return [
        create_list_documents_tool(tenant_id),
        create_search_tool(tenant_id),
        create_document_search_tool(tenant_id),
    ]
