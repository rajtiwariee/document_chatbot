"""
Multi-format document text extractor.

Supports: PDF, DOCX, PPTX, XLSX, and images (with OCR).
Uses the 'unstructured' library for structured extraction.
"""
import logging
from pathlib import Path
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ExtractedPage:
    """A single page/sheet/slide of extracted content."""
    page_number: int
    text: str
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


class DocumentExtractor:
    """
    Extract text from multiple document formats.

    Usage:
        extractor = DocumentExtractor()
        result = extractor.extract("/path/to/doc.pdf", "pdf")
    """

    SUPPORTED_TYPES = {"pdf", "docx", "pptx", "xlsx", "image", "other"}

    def extract(self, file_path: str, file_type: str) -> ExtractedDocument:
        """
        Extract text from a document.

        Args:
            file_path: Absolute path to the file
            file_type: One of: pdf, docx, pptx, xlsx, image, other

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
            # Fallback: if it's "other" but not in supported (shouldn't happen if we add it)
            # or if it's some unknown type, we might try to read as text if it's "other"
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
        """Extract text from PDF using unstructured."""
        from unstructured.partition.pdf import partition_pdf

        elements = partition_pdf(str(path), strategy="auto")
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
        """Extract text from Excel spreadsheets."""
        from unstructured.partition.xlsx import partition_xlsx

        elements = partition_xlsx(str(path))
        return self._elements_to_document(path, "xlsx", elements)

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
            # Try latin-1 fallback
            text = path.read_text(encoding="latin-1")
            
        # Treat entire file as page 1
        page = ExtractedPage(
            page_number=1,
            text=text,
            metadata={"source_file": path.name}
        )
        
        doc = ExtractedDocument(
            filename=path.name,
            file_type="other",
            pages=[page],
            total_text=text,
            metadata={"source_path": str(path), "page_count": 1}
        )
        return doc
    
    def _elements_to_document(
        self, path: Path, file_type: str, elements: list
    ) -> ExtractedDocument:
        """Convert unstructured elements to our ExtractedDocument format."""
        pages: dict[int, list[str]] = {}

        for element in elements:
            # Get page number from metadata (default to 1)
            page_num = getattr(element.metadata, "page_number", 1) or 1
            if page_num not in pages:
                pages[page_num] = []
            pages[page_num].append(str(element))

        extracted_pages = [
            ExtractedPage(
                page_number=num,
                text="\n".join(texts),
                metadata={"source_file": path.name},
            )
            for num, texts in sorted(pages.items())
        ]

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
