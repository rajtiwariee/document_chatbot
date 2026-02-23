"""
Semantic text chunking for optimal retrieval.

Splits extracted documents into overlapping chunks that maintain
context for accurate vector search and LLM reasoning.

Table-aware: tables are kept as whole chunks (never split mid-row).
Very large tables are split by rows with the header repeated.
"""
import logging
import uuid
from dataclasses import dataclass, field

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.document_processing.extractor import ExtractedDocument, ElementType

logger = logging.getLogger(__name__)

# Maximum size for a single table chunk before splitting by rows
MAX_TABLE_CHUNK_SIZE = 3000


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
        """Rough token estimate (1 token ~ 4 chars)."""
        return len(self.text) // 4


def _split_large_table(table_text: str, max_size: int = MAX_TABLE_CHUNK_SIZE) -> list[str]:
    """
    Split a large markdown table by rows, repeating the header in each sub-chunk.

    The header is the first two lines (header row + separator row).
    """
    lines = table_text.strip().split("\n")
    if len(lines) <= 2:
        return [table_text]

    header = lines[0] + "\n" + lines[1]
    data_rows = lines[2:]
    header_size = len(header) + 1  # +1 for newline

    sub_chunks = []
    current_rows: list[str] = []
    current_size = header_size

    for row in data_rows:
        row_size = len(row) + 1
        if current_size + row_size > max_size and current_rows:
            sub_chunks.append(header + "\n" + "\n".join(current_rows))
            current_rows = []
            current_size = header_size
        current_rows.append(row)
        current_size += row_size

    if current_rows:
        sub_chunks.append(header + "\n" + "\n".join(current_rows))

    return sub_chunks


class SemanticChunker:
    """
    Chunk documents with semantic awareness using recursive character splitting.

    When structured elements are available, tables are kept whole (or split by
    rows for very large tables) and text elements are batched through the
    standard RecursiveCharacterTextSplitter.
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
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

        If structured elements are available, uses element-type-aware chunking.
        Otherwise falls back to flat text splitting.
        """
        chunks: list[Chunk] = []
        chunk_index = 0

        for page in document.pages:
            if not page.text or not page.text.strip():
                continue

            # Use element-aware chunking if structured elements exist
            if page.elements:
                page_chunks = self._chunk_elements(
                    page.elements, page.page_number, document, document_id, tenant_id, chunk_index
                )
                chunks.extend(page_chunks)
                chunk_index += len(page_chunks)
            else:
                # Fallback: flat text splitting
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

    def _chunk_elements(
        self,
        elements: list,
        page_number: int,
        document: ExtractedDocument,
        document_id: str,
        tenant_id: str,
        start_index: int,
    ) -> list[Chunk]:
        """Process elements by type: keep tables whole, batch text through splitter."""
        chunks: list[Chunk] = []
        chunk_index = start_index
        text_buffer: list[str] = []
        current_section: str | None = None

        def _flush_text_buffer():
            nonlocal chunk_index
            if not text_buffer:
                return
            combined = "\n".join(text_buffer)
            text_buffer.clear()

            splits = self.splitter.split_text(combined)
            for split_text in splits:
                if not split_text.strip():
                    continue
                chunks.append(
                    Chunk(
                        text=split_text,
                        document_id=document_id,
                        tenant_id=tenant_id,
                        page_number=page_number,
                        chunk_index=chunk_index,
                        metadata={
                            "source_file": document.filename,
                            "file_type": document.file_type,
                            "page_number": page_number,
                            "chunk_index": chunk_index,
                            "element_type": ElementType.TEXT.value,
                            "section_header": current_section,
                        },
                    )
                )
                chunk_index += 1

        for element in elements:
            # Track the current section heading as we iterate
            if element.element_type == ElementType.TITLE:
                current_section = element.text

            if element.element_type == ElementType.TABLE:
                # Flush any accumulated text first
                _flush_text_buffer()

                section = element.metadata.get("section_header") or current_section

                # Check if table is too large
                if len(element.text) > MAX_TABLE_CHUNK_SIZE:
                    sub_chunks = _split_large_table(element.text)
                else:
                    sub_chunks = [element.text]

                sheet_name = element.metadata.get("sheet_name", "")

                for table_text in sub_chunks:
                    if not table_text.strip():
                        continue

                    # Prepend context so the table is self-describing
                    context_prefix = ""
                    if section:
                        context_prefix = f"## {section}\n\n"

                    chunks.append(
                        Chunk(
                            text=context_prefix + table_text,
                            document_id=document_id,
                            tenant_id=tenant_id,
                            page_number=page_number,
                            chunk_index=chunk_index,
                            metadata={
                                "source_file": document.filename,
                                "file_type": document.file_type,
                                "page_number": page_number,
                                "chunk_index": chunk_index,
                                "element_type": ElementType.TABLE.value,
                                "section_header": section,
                                "sheet_name": sheet_name,
                            },
                        )
                    )
                    chunk_index += 1
            else:
                # Accumulate text/title/list_item in buffer
                text_buffer.append(element.text)

        # Flush remaining text
        _flush_text_buffer()

        return chunks
