"""
Semantic text chunking for optimal retrieval.

Splits extracted documents into overlapping chunks that maintain
context for accurate vector search and LLM reasoning.
"""
import logging
import uuid
from dataclasses import dataclass, field

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.document_processing.extractor import ExtractedDocument

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """A single text chunk ready for embedding."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    text: str = ""
    document_id: str = ""
    tenant_id: str = ""
    page_number: int | None = None
    chunk_index: int = 0
    metadata: dict = field(default_factory=dict)

    @property
    def token_estimate(self) -> int:
        """Rough token estimate (1 token ≈ 4 chars)."""
        return len(self.text) // 4


class SemanticChunker:
    """
    Chunk documents with semantic awareness using recursive character splitting.

    Default settings:
    - chunk_size: 1000 characters (~250 tokens)
    - chunk_overlap: 200 characters (20% overlap for context continuity)
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

    def chunk(
        self,
        document: ExtractedDocument,
        document_id: str,
        tenant_id: str,
    ) -> list[Chunk]:
        """
        Split an extracted document into retrievable chunks with metadata.

        Each chunk preserves:
        - Source document ID and tenant ID
        - Original page number
        - Chunk index for ordering
        - Source filename in metadata
        """
        chunks: list[Chunk] = []
        chunk_index = 0

        for page in document.pages:
            if not page.text or not page.text.strip():
                continue

            # Split the page text
            splits = self.splitter.split_text(page.text)

            for split_text in splits:
                if not split_text.strip():
                    continue

                chunks.append(
                    Chunk(
                        text=split_text,
                        document_id=document_id,
                        tenant_id=tenant_id,
                        page_number=page.page_number,
                        chunk_index=chunk_index,
                        metadata={
                            "source_file": document.filename,
                            "file_type": document.file_type,
                            "page_number": page.page_number,
                            "chunk_index": chunk_index,
                        },
                    )
                )
                chunk_index += 1

        logger.info(
            f"Chunked {document.filename}: {len(chunks)} chunks "
            f"from {document.page_count} pages"
        )
        return chunks
