"""
Multi-format document text extractor.

Supports: PDF, DOCX, PPTX, XLSX, CSV, and images (with OCR).
Uses the 'unstructured' library for structured extraction with
special handling for tables to preserve row/column structure.
"""
import csv
import io
import logging
from enum import Enum
from html.parser import HTMLParser
from pathlib import Path
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class ElementType(str, Enum):
    """Types of content elements extracted from documents."""
    TEXT = "text"
    TABLE = "table"
    TITLE = "title"
    LIST_ITEM = "list_item"


@dataclass
class ContentElement:
    """A typed, structured content element from a document."""
    element_type: ElementType
    text: str
    metadata: dict = field(default_factory=dict)


@dataclass
class ExtractedPage:
    """A single page/sheet/slide of extracted content."""
    page_number: int
    text: str
    elements: list[ContentElement] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class ExtractedDocument:
    """Result of document extraction."""
    filename: str
    file_type: str
    pages: list[ExtractedPage] = field(default_factory=list)
    total_text: str = ""
    metadata: dict = field(default_factory=dict)
    error: str | None = None

    @property
    def page_count(self) -> int:
        return len(self.pages)

    def full_text(self) -> str:
        """Get all text combined."""
        if self.total_text:
            return self.total_text
        return "\n\n".join(p.text for p in self.pages if p.text)


class _HTMLTableParser(HTMLParser):
    """Parse HTML <table> into a list of rows (each row is a list of cell strings)."""

    def __init__(self):
        super().__init__()
        self.rows: list[list[str]] = []
        self._current_row: list[str] = []
        self._current_cell: list[str] = []
        self._in_cell = False

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self._current_row = []
        elif tag in ("td", "th"):
            self._current_cell = []
            self._in_cell = True

    def handle_endtag(self, tag):
        if tag in ("td", "th"):
            self._in_cell = False
            self._current_row.append("".join(self._current_cell).strip())
        elif tag == "tr":
            if self._current_row:
                self.rows.append(self._current_row)

    def handle_data(self, data):
        if self._in_cell:
            self._current_cell.append(data)


def _html_table_to_markdown(html: str) -> str:
    """Convert an HTML <table> string to a markdown pipe-delimited table."""
    parser = _HTMLTableParser()
    parser.feed(html)
    rows = parser.rows

    if not rows:
        return ""

    # Determine max columns for alignment
    max_cols = max(len(r) for r in rows)

    # Normalize all rows to same column count
    normalized = []
    for row in rows:
        padded = row + [""] * (max_cols - len(row))
        normalized.append(padded)

    lines = []
    # Header row
    header = normalized[0]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join("---" for _ in header) + " |")

    # Data rows
    for row in normalized[1:]:
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def _rows_to_markdown(rows: list[list[str]], sheet_name: str | None = None) -> str:
    """Convert a list of rows to a markdown table with optional sheet name header."""
    if not rows:
        return ""

    max_cols = max(len(r) for r in rows)
    normalized = []
    for row in rows:
        padded = [str(cell) if cell is not None else "" for cell in row] + [""] * (max_cols - len(row))
        normalized.append(padded)

    lines = []
    if sheet_name:
        lines.append(f"### Sheet: {sheet_name}\n")

    header = normalized[0]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join("---" for _ in header) + " |")

    for row in normalized[1:]:
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


class DocumentExtractor:
    """
    Extract text from multiple document formats.

    Usage:
        extractor = DocumentExtractor()
        result = extractor.extract("/path/to/doc.pdf", "pdf")
    """

    SUPPORTED_TYPES = {"pdf", "docx", "pptx", "xlsx", "csv", "image", "other"}

    def extract(self, file_path: str, file_type: str) -> ExtractedDocument:
        """
        Extract text from a document.

        Args:
            file_path: Absolute path to the file
            file_type: One of: pdf, docx, pptx, xlsx, csv, image, other

        Returns:
            ExtractedDocument with extracted text and metadata
        """
        path = Path(file_path)
        if not path.exists():
            return ExtractedDocument(
                filename=path.name,
                file_type=file_type,
                error=f"File not found: {file_path}",
            )

        if file_type not in self.SUPPORTED_TYPES:
            return ExtractedDocument(
                filename=path.name,
                file_type=file_type,
                error=f"Unsupported file type: {file_type}",
            )

        try:
            method = getattr(self, f"_extract_{file_type}")
            return method(path)
        except Exception as e:
            logger.exception(f"Error extracting {file_path}")
            return ExtractedDocument(
                filename=path.name,
                file_type=file_type,
                error=str(e),
            )

    def _extract_pdf(self, path: Path) -> ExtractedDocument:
        """Extract text from PDF using unstructured with hi_res for table detection."""
        from unstructured.partition.pdf import partition_pdf

        elements = partition_pdf(
            str(path),
            strategy="hi_res",
            infer_table_structure=True,
        )
        return self._elements_to_document(path, "pdf", elements)

    def _extract_docx(self, path: Path) -> ExtractedDocument:
        """Extract text from Word documents."""
        from unstructured.partition.docx import partition_docx

        elements = partition_docx(str(path))
        return self._elements_to_document(path, "docx", elements)

    def _extract_pptx(self, path: Path) -> ExtractedDocument:
        """Extract text from PowerPoint."""
        from unstructured.partition.pptx import partition_pptx

        elements = partition_pptx(str(path))
        return self._elements_to_document(path, "pptx", elements)

    def _extract_xlsx(self, path: Path) -> ExtractedDocument:
        """Extract text from Excel spreadsheets using openpyxl for structure preservation."""
        import openpyxl

        wb = openpyxl.load_workbook(str(path), data_only=True)
        pages = []

        for sheet_idx, sheet_name in enumerate(wb.sheetnames, start=1):
            ws = wb[sheet_name]
            rows = []
            for row in ws.iter_rows(values_only=True):
                # Skip completely empty rows
                if any(cell is not None for cell in row):
                    rows.append([str(cell) if cell is not None else "" for cell in row])

            if not rows:
                continue

            md_table = _rows_to_markdown(rows, sheet_name=sheet_name)
            plain_text = md_table

            elements = [
                ContentElement(
                    element_type=ElementType.TABLE,
                    text=md_table,
                    metadata={"sheet_name": sheet_name},
                )
            ]

            pages.append(ExtractedPage(
                page_number=sheet_idx,
                text=plain_text,
                elements=elements,
                metadata={"source_file": path.name, "sheet_name": sheet_name},
            ))

        wb.close()

        doc = ExtractedDocument(
            filename=path.name,
            file_type="xlsx",
            pages=pages,
            metadata={
                "source_path": str(path),
                "page_count": len(pages),
            },
        )
        doc.total_text = doc.full_text()

        logger.info(
            f"Extracted {path.name}: {len(pages)} sheets, "
            f"{len(doc.total_text)} chars"
        )
        return doc

    def _extract_csv(self, path: Path) -> ExtractedDocument:
        """Extract text from CSV files, preserving as markdown table."""
        text = path.read_text(encoding="utf-8-sig")
        reader = csv.reader(io.StringIO(text))
        rows = [row for row in reader if any(cell.strip() for cell in row)]

        if not rows:
            return ExtractedDocument(
                filename=path.name,
                file_type="csv",
                error="CSV file is empty",
            )

        md_table = _rows_to_markdown(rows)

        elements = [
            ContentElement(
                element_type=ElementType.TABLE,
                text=md_table,
                metadata={"row_count": len(rows)},
            )
        ]

        page = ExtractedPage(
            page_number=1,
            text=md_table,
            elements=elements,
            metadata={"source_file": path.name},
        )

        doc = ExtractedDocument(
            filename=path.name,
            file_type="csv",
            pages=[page],
            metadata={"source_path": str(path), "page_count": 1},
        )
        doc.total_text = md_table

        logger.info(f"Extracted {path.name}: {len(rows)} rows, {len(md_table)} chars")
        return doc

    def _extract_image(self, path: Path) -> ExtractedDocument:
        """Extract text from images using OCR."""
        from unstructured.partition.image import partition_image

        elements = partition_image(str(path), strategy="ocr_only")
        return self._elements_to_document(path, "image", elements)

    def _extract_other(self, path: Path) -> ExtractedDocument:
        """Extract text from plain text files."""
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = path.read_text(encoding="latin-1")

        page = ExtractedPage(
            page_number=1,
            text=text,
            elements=[ContentElement(element_type=ElementType.TEXT, text=text)],
            metadata={"source_file": path.name},
        )

        doc = ExtractedDocument(
            filename=path.name,
            file_type="other",
            pages=[page],
            total_text=text,
            metadata={"source_path": str(path), "page_count": 1},
        )
        return doc

    def _classify_element(self, element) -> tuple[ElementType, str]:
        """Classify an unstructured element and extract its text with structure."""
        element_class = type(element).__name__

        if element_class in ("Table", "TableChunk"):
            # Try to get HTML representation for accurate table conversion
            html = getattr(element.metadata, "text_as_html", None)
            if html:
                md_table = _html_table_to_markdown(html)
                if md_table:
                    return ElementType.TABLE, md_table
            # Fallback: wrap in fenced code block to preserve spacing
            return ElementType.TABLE, f"```\n{str(element)}\n```"

        if element_class == "Title":
            return ElementType.TITLE, str(element)

        if element_class == "ListItem":
            return ElementType.LIST_ITEM, f"- {str(element)}"

        # NarrativeText, UncategorizedText, etc.
        return ElementType.TEXT, str(element)

    def _elements_to_document(
        self, path: Path, file_type: str, elements: list
    ) -> ExtractedDocument:
        """Convert unstructured elements to our ExtractedDocument format with structure."""
        pages: dict[int, list[ContentElement]] = {}
        current_section: str | None = None

        for element in elements:
            page_num = getattr(element.metadata, "page_number", 1) or 1
            if page_num not in pages:
                pages[page_num] = []

            el_type, text = self._classify_element(element)

            # Track section headers
            if el_type == ElementType.TITLE:
                current_section = text

            content_el = ContentElement(
                element_type=el_type,
                text=text,
                metadata={"section_header": current_section},
            )
            pages[page_num].append(content_el)

        extracted_pages = []
        for num, elements_list in sorted(pages.items()):
            # Build plain text from elements
            text_parts = []
            for el in elements_list:
                if el.element_type == ElementType.TITLE:
                    text_parts.append(f"\n## {el.text}\n")
                else:
                    text_parts.append(el.text)

            page_text = "\n".join(text_parts)

            extracted_pages.append(ExtractedPage(
                page_number=num,
                text=page_text,
                elements=elements_list,
                metadata={"source_file": path.name},
            ))

        doc = ExtractedDocument(
            filename=path.name,
            file_type=file_type,
            pages=extracted_pages,
            metadata={
                "source_path": str(path),
                "page_count": len(extracted_pages),
            },
        )
        doc.total_text = doc.full_text()

        logger.info(
            f"Extracted {path.name}: {len(extracted_pages)} pages, "
            f"{len(doc.total_text)} chars"
        )
        return doc
