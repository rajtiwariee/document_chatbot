"""
Debug artifact saver for the document processing pipeline.

Saves intermediate outputs (extraction, tables, chunks, summary) to disk
so we can diagnose truncation and data-loss issues at each pipeline stage.
"""
import json
import logging
import os
from pathlib import Path

from app.document_processing.chunker import Chunk
from app.document_processing.extractor import ElementType, ExtractedDocument

logger = logging.getLogger(__name__)

DEFAULT_BASE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "logs",
    "processing_artifacts",
)


class ProcessingArtifactSaver:
    """Saves debug artifacts after each stage of document processing."""

    def __init__(
        self,
        tenant_id: str,
        document_id: str,
        base_dir: str = DEFAULT_BASE_DIR,
    ):
        self.tenant_id = tenant_id
        self.document_id = document_id
        self.artifact_dir = Path(base_dir) / tenant_id / document_id
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

    def _write(self, filename: str, content: str) -> None:
        path = self.artifact_dir / filename
        path.write_text(content, encoding="utf-8")
        logger.info(f"[{self.document_id}] Saved artifact: {path}")

    # ── Extraction ───────────────────────────────────────────────────

    def save_extraction(self, extracted: ExtractedDocument) -> None:
        """Save full extraction output — raw VLM text verbatim, no truncation."""
        total_chars = sum(len(p.text) for p in extracted.pages)
        lines = [
            f"# Extraction Output: {extracted.filename}",
            f"- File Type: {extracted.file_type}",
            f"- Pipeline: {extracted.metadata.get('pipeline', 'unknown')}",
            f"- PDF Type: {extracted.metadata.get('pdf_type', 'n/a')}",
            f"- Pages: {extracted.page_count}",
            f"- Total Chars: {total_chars}",
            "",
        ]

        for page in extracted.pages:
            page_chars = len(page.text)
            num_elements = len(page.elements)
            lines.append(f"## Page {page.page_number} ({page_chars} chars, {num_elements} elements)")
            lines.append("")

            if page.elements:
                lines.append("### Elements")
                for i, elem in enumerate(page.elements, 1):
                    lines.append(
                        f"#### Element {i}: {elem.element_type.value.upper()} ({len(elem.text)} chars)"
                    )
                    lines.append("```")
                    lines.append(elem.text)
                    lines.append("```")
                    lines.append("")
            else:
                lines.append("### Raw Text")
                lines.append("```")
                lines.append(page.text)
                lines.append("```")
                lines.append("")

        self._write("extraction.md", "\n".join(lines))

    # ── Tables ───────────────────────────────────────────────────────

    def save_tables(self, extracted: ExtractedDocument) -> None:
        """Save only table elements for quick inspection."""
        table_blocks: list[str] = []
        table_count = 0

        for page in extracted.pages:
            for elem in page.elements:
                if elem.element_type == ElementType.TABLE:
                    table_count += 1
                    table_blocks.append(
                        f"## Table {table_count} (Page {page.page_number}, {len(elem.text)} chars)"
                    )
                    table_blocks.append("")
                    table_blocks.append(elem.text)
                    table_blocks.append("")

        lines = [
            f"# Tables: {extracted.filename}",
            f"- Total Tables: {table_count}",
            "",
        ]
        if table_count == 0:
            lines.append("_No table elements found._")
        else:
            lines.extend(table_blocks)

        self._write("tables.md", "\n".join(lines))

    # ── Chunks ───────────────────────────────────────────────────────

    def save_chunks(self, chunks: list[Chunk]) -> None:
        """Save all chunks with full text."""
        lines = [f"# Chunks: {len(chunks)} total", ""]

        for i, chunk in enumerate(chunks):
            chunk_type = chunk.metadata.get("type", "unknown")
            lines.append(
                f"## Chunk {i} (Page {chunk.page_number}, {len(chunk.text)} chars, type: {chunk_type})"
            )
            lines.append("```")
            lines.append(chunk.text)
            lines.append("```")
            lines.append("")

        self._write("chunks.md", "\n".join(lines))

    # ── Summary ──────────────────────────────────────────────────────

    def save_summary(
        self,
        extracted: ExtractedDocument,
        chunks: list[Chunk],
        num_vectors: int,
    ) -> None:
        """Save a concise stats overview."""
        total_extract_chars = sum(len(p.text) for p in extracted.pages)
        total_chunk_chars = sum(len(c.text) for c in chunks)

        lines = [
            f"# Processing Summary: {extracted.filename}",
            "",
            "## Extraction",
            f"- File Type: {extracted.file_type}",
            f"- Pipeline: {extracted.metadata.get('pipeline', 'unknown')}",
            f"- PDF Type: {extracted.metadata.get('pdf_type', 'n/a')}",
            f"- Pages: {extracted.page_count}",
            f"- Total Extracted Chars: {total_extract_chars}",
            "",
            "### Per-Page Stats",
        ]
        for page in extracted.pages:
            num_tables = sum(1 for e in page.elements if e.element_type == ElementType.TABLE)
            lines.append(
                f"- Page {page.page_number}: {len(page.text)} chars, "
                f"{len(page.elements)} elements ({num_tables} tables)"
            )
        lines += [
            "",
            "## Chunking",
            f"- Chunks: {len(chunks)}",
            f"- Total Chunk Chars: {total_chunk_chars}",
            f"- Avg Chunk Size: {total_chunk_chars // max(len(chunks), 1)} chars",
            "",
            "## Indexing",
            f"- Vectors Stored: {num_vectors}",
        ]

        self._write("summary.md", "\n".join(lines))

    # ── Report (JSON) ────────────────────────────────────────────────

    def save_report(
        self,
        extracted: ExtractedDocument,
        chunks: list[Chunk],
        num_vectors: int,
    ) -> None:
        """Save a full JSON dump for programmatic analysis."""
        report = {
            "filename": extracted.filename,
            "file_type": extracted.file_type,
            "metadata": extracted.metadata,
            "page_count": extracted.page_count,
            "total_extract_chars": sum(len(p.text) for p in extracted.pages),
            "pages": [
                {
                    "page_number": p.page_number,
                    "text_length": len(p.text),
                    "text": p.text,
                    "metadata": p.metadata,
                    "elements": [
                        {
                            "type": e.element_type.value,
                            "text_length": len(e.text),
                            "text": e.text,
                            "metadata": e.metadata,
                        }
                        for e in p.elements
                    ],
                }
                for p in extracted.pages
            ],
            "chunks": [
                {
                    "index": c.chunk_index,
                    "page_number": c.page_number,
                    "text_length": len(c.text),
                    "text": c.text,
                    "metadata": c.metadata,
                }
                for c in chunks
            ],
            "num_chunks": len(chunks),
            "total_chunk_chars": sum(len(c.text) for c in chunks),
            "num_vectors": num_vectors,
        }
        self._write("report.json", json.dumps(report, indent=2, default=str))

    # ── Convenience ──────────────────────────────────────────────────

    def save_all(
        self,
        extracted: ExtractedDocument,
        chunks: list[Chunk],
        num_vectors: int,
    ) -> None:
        """Save all artifacts. Each save is independent — failures don't cascade."""
        for fn in [
            lambda: self.save_extraction(extracted),
            lambda: self.save_tables(extracted),
            lambda: self.save_chunks(chunks),
            lambda: self.save_summary(extracted, chunks, num_vectors),
            lambda: self.save_report(extracted, chunks, num_vectors),
        ]:
            try:
                fn()
            except Exception as e:
                logger.warning(f"[{self.document_id}] Artifact save failed: {e}")
