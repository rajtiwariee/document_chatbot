"""
Multi-format document text extractor.

Supports: PDF, DOCX, PPTX, XLSX, CSV, and images (with OCR).
Uses the 'unstructured' library for structured extraction with
special handling for tables to preserve row/column structure.
"""
import asyncio
import csv
import io
import logging
import os
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


def _rows_to_markdown(rows: list[list[str]]) -> str:
    """Convert a list of rows to a markdown pipe-delimited table."""
    if not rows:
        return ""

    max_cols = max(len(r) for r in rows)
    normalized = []
    for row in rows:
        padded = [str(cell) if cell is not None else "" for cell in row] + [""] * (max_cols - len(row))
        normalized.append(padded)

    lines = []
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

        # With vision-enhanced table extraction for PDFs:
        extractor = DocumentExtractor(
            use_vision_tables=True,
            vision_backend=get_vision_backend(),
        )
    """

    SUPPORTED_TYPES = {"pdf", "docx", "pptx", "xlsx", "csv", "image", "other"}

    # PyMuPDF fallback image extraction — size filters for decorative junk
    _PYMUPDF_IMG_MIN_WIDTH = 50
    _PYMUPDF_IMG_MIN_HEIGHT = 50
    _PYMUPDF_IMG_MIN_AREA = 10_000

    def __init__(self, use_vision_tables: bool = False, vision_backend=None):
        self._use_vision_tables = use_vision_tables
        self._vision_backend = vision_backend

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
        if self._use_vision_tables and self._vision_backend:
            return self._extract_pdf_with_vision(path)

        from unstructured.partition.pdf import partition_pdf

        elements = partition_pdf(
            str(path),
            strategy="hi_res",
            infer_table_structure=True,
        )
        return self._elements_to_document(path, "pdf", elements)

    def _extract_pdf_with_vision(self, path: Path) -> ExtractedDocument:
        """
        Extract PDF with vision-enhanced table extraction.

        Routes to the appropriate pipeline based on PDF type:
        - Scanned PDFs → VLM full-page extraction (skip unstructured).
        - Digital/mixed PDFs → Unstructured + VLM table/image enhancement.
        """
        from app.document_processing.page_renderer import detect_pdf_type

        pdf_type, page_types = detect_pdf_type(str(path))
        logger.info("PDF %s detected as '%s'", path.name, pdf_type)

        if pdf_type == "scanned":
            return self._extract_scanned_pdf_with_vision(path)
        elif pdf_type == "mixed":
            return self._extract_mixed_pdf_with_vision(path, page_types)
        else:
            return self._extract_digital_pdf_with_vision(path)

    def _extract_scanned_pdf_with_vision(self, path: Path) -> ExtractedDocument:
        """
        Extract a fully scanned PDF by rendering each page and sending to VLM.

        Each page is rendered at 300 DPI and the VLM extracts all content
        (text, tables, figures) in a single pass per page.
        """
        import fitz
        from app.document_processing.page_renderer import render_page_to_image

        logger.info("Scanned PDF pipeline for %s", path.name)

        doc = fitz.open(str(path))
        page_count = len(doc)
        doc.close()

        temp_files: list[str] = []
        extracted_pages: list[ExtractedPage] = []

        try:
            loop = asyncio.new_event_loop()
            try:
                for page_num in range(1, page_count + 1):
                    # Render at 300 DPI for better OCR quality on scans
                    page_img = render_page_to_image(str(path), page_num, dpi=300)
                    temp_files.append(page_img)

                    try:
                        page_text = loop.run_until_complete(
                            self._vision_backend.extract_page(page_img)
                        )
                    except Exception:
                        logger.exception(
                            "VLM page extraction failed on page %d of %s",
                            page_num, path.name,
                        )
                        page_text = ""

                    # Parse VLM output into typed content elements
                    elements = self._parse_vlm_page_output(page_text)

                    extracted_pages.append(ExtractedPage(
                        page_number=page_num,
                        text=page_text,
                        elements=elements,
                        metadata={"source_file": path.name, "pipeline": "scanned_vlm"},
                    ))

                    logger.info(
                        "Scanned page %d/%d extracted (%d chars, %d elements)",
                        page_num, page_count, len(page_text), len(elements),
                    )
            finally:
                loop.close()

            doc_result = ExtractedDocument(
                filename=path.name,
                file_type="pdf",
                pages=extracted_pages,
                metadata={
                    "source_path": str(path),
                    "page_count": len(extracted_pages),
                    "pdf_type": "scanned",
                    "pipeline": "scanned_vlm",
                },
            )
            doc_result.total_text = doc_result.full_text()

            logger.info(
                "Scanned PDF extraction complete: %s — %d pages, %d chars",
                path.name, len(extracted_pages), len(doc_result.total_text),
            )
            return doc_result

        finally:
            for tmp_path in temp_files:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    def _extract_mixed_pdf_with_vision(
        self, path: Path, page_types: dict[int, str]
    ) -> ExtractedDocument:
        """
        Extract a mixed PDF with per-page routing.

        Scanned pages → VLM full-page extraction (replaces unstructured output).
        Digital pages  → Unstructured + VLM table/image enhancements.
        """
        from unstructured.partition.pdf import partition_pdf
        from app.document_processing.page_renderer import (
            render_page_to_image,
            crop_region_from_page,
        )

        scanned_pages = {p for p, t in page_types.items() if t == "scanned"}
        digital_pages = {p for p, t in page_types.items() if t == "digital"}
        logger.info(
            "Mixed PDF pipeline for %s — %d scanned pages %s, %d digital pages %s",
            path.name,
            len(scanned_pages), sorted(scanned_pages),
            len(digital_pages), sorted(digital_pages),
        )

        # Run unstructured on the full PDF (needed for digital pages;
        # scanned page results will be discarded and replaced with VLM output)
        elements = partition_pdf(
            str(path),
            strategy="hi_res",
            infer_table_structure=True,
            extract_image_block_to_payload=True,
        )

        # Partition unstructured elements by page
        elements_by_page: dict[int, list] = {}
        for el in elements:
            pg = getattr(el.metadata, "page_number", 1) or 1
            elements_by_page.setdefault(pg, []).append(el)

        temp_files: list[str] = []
        # Will hold final ExtractedPage objects keyed by page number
        final_pages: dict[int, ExtractedPage] = {}

        try:
            loop = asyncio.new_event_loop()
            try:
                # --- Scanned pages: VLM full-page extraction ---
                for page_num in sorted(scanned_pages):
                    page_img = render_page_to_image(str(path), page_num, dpi=300)
                    temp_files.append(page_img)

                    try:
                        page_text = loop.run_until_complete(
                            self._vision_backend.extract_page(page_img)
                        )
                    except Exception:
                        logger.exception(
                            "VLM page extraction failed on scanned page %d of %s",
                            page_num, path.name,
                        )
                        page_text = ""

                    vlm_elements = self._parse_vlm_page_output(page_text)
                    final_pages[page_num] = ExtractedPage(
                        page_number=page_num,
                        text=page_text,
                        elements=vlm_elements,
                        metadata={
                            "source_file": path.name,
                            "pipeline": "scanned_vlm",
                            "page_type": "scanned",
                        },
                    )
                    logger.info(
                        "VLM page extraction on scanned page %d (%d chars, %d elements)",
                        page_num, len(page_text), len(vlm_elements),
                    )

                # --- Digital pages: enhance with VLM tables + image captions ---
                for page_num in sorted(digital_pages):
                    page_els = elements_by_page.get(page_num, [])

                    # Vision table extraction for Table elements on this page
                    table_els = [
                        el for el in page_els
                        if type(el).__name__ in ("Table", "TableChunk")
                    ]
                    if table_els:
                        page_img = render_page_to_image(str(path), page_num)
                        temp_files.append(page_img)

                        for el in table_els:
                            coords = getattr(el.metadata, "coordinates", None)
                            if not coords:
                                continue

                            crop_path = crop_region_from_page(
                                page_img, coords.points,
                                coords.system.width, coords.system.height,
                            )
                            temp_files.append(crop_path)

                            try:
                                md_table = loop.run_until_complete(
                                    self._vision_backend.extract_table(crop_path)
                                )
                                if md_table and md_table.strip() != "NO_TABLE_FOUND":
                                    el._vision_table_text = md_table
                                    logger.info(
                                        "Vision table extracted on digital page %d (%d chars)",
                                        page_num, len(md_table),
                                    )
                            except Exception:
                                logger.exception(
                                    "Vision table extraction failed on digital page %d",
                                    page_num,
                                )

                    # Vision captions for Image elements on this page
                    for el in page_els:
                        if type(el).__name__ == "Image":
                            image_b64 = getattr(el.metadata, "image_base64", None)
                            if image_b64:
                                import base64
                                import tempfile as tf

                                img_bytes = base64.b64decode(image_b64)
                                tmp = tf.NamedTemporaryFile(
                                    suffix=".png", prefix="img_element_", delete=False
                                )
                                tmp.write(img_bytes)
                                tmp.close()
                                temp_files.append(tmp.name)

                                try:
                                    caption = loop.run_until_complete(
                                        self._vision_backend.caption_image(tmp.name)
                                    )
                                    el._vision_caption = caption
                                except Exception:
                                    logger.exception(
                                        "Vision captioning failed for image on digital page %d",
                                        page_num,
                                    )

                # --- PyMuPDF fallback for missed images (digital pages only) ---
                from types import SimpleNamespace

                pymupdf_missed = self._collect_pymupdf_images(str(path))

                for pg_num, img_list in pymupdf_missed.items():
                    if pg_num in scanned_pages:
                        # Scanned pages fully covered by VLM — skip fallback
                        for tmp_path in img_list:
                            temp_files.append(tmp_path)
                        continue

                    for tmp_path in img_list:
                        temp_files.append(tmp_path)
                        try:
                            caption = loop.run_until_complete(
                                self._vision_backend.caption_image(tmp_path)
                            )
                        except Exception:
                            logger.exception(
                                "Vision captioning failed for PyMuPDF image on page %d",
                                pg_num,
                            )
                            continue

                        _SyntheticImage = type("Image", (), {})
                        synthetic_el = _SyntheticImage()
                        synthetic_el.metadata = SimpleNamespace(page_number=pg_num)
                        synthetic_el._vision_caption = caption
                        elements_by_page.setdefault(pg_num, []).append(synthetic_el)
            finally:
                loop.close()

            # --- Build digital pages through _elements_to_document path ---
            # Collect all digital-page elements back into a flat list for conversion
            digital_elements = []
            for pg_num in sorted(digital_pages):
                digital_elements.extend(elements_by_page.get(pg_num, []))

            if digital_elements:
                digital_doc = self._elements_to_document(path, "pdf", digital_elements)
                for page in digital_doc.pages:
                    page.metadata["pipeline"] = "digital_unstructured"
                    page.metadata["page_type"] = "digital"
                    final_pages[page.page_number] = page

            # --- Merge into single ExtractedDocument ---
            all_pages = [final_pages[n] for n in sorted(final_pages)]

            doc_result = ExtractedDocument(
                filename=path.name,
                file_type="pdf",
                pages=all_pages,
                metadata={
                    "source_path": str(path),
                    "page_count": len(all_pages),
                    "pdf_type": "mixed",
                    "pipeline": "mixed_per_page",
                    "scanned_pages": sorted(scanned_pages),
                    "digital_pages": sorted(digital_pages),
                },
            )
            doc_result.total_text = doc_result.full_text()

            logger.info(
                "Mixed PDF extraction complete: %s — %d pages (%d scanned, %d digital), %d chars",
                path.name, len(all_pages), len(scanned_pages),
                len(digital_pages), len(doc_result.total_text),
            )
            return doc_result

        finally:
            for tmp_path in temp_files:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    @staticmethod
    def _parse_vlm_page_output(text: str) -> list[ContentElement]:
        """
        Parse VLM full-page output into typed ContentElement instances.

        - Lines starting with | → TABLE elements (grouped into contiguous blocks).
        - Lines starting with # → TITLE elements.
        - Everything else → TEXT elements (grouped into contiguous blocks).
        """
        if not text or not text.strip():
            return []

        elements: list[ContentElement] = []
        lines = text.split("\n")

        current_type: ElementType | None = None
        current_lines: list[str] = []

        def _flush():
            nonlocal current_type, current_lines
            if current_type is None or not current_lines:
                current_lines = []
                return
            block = "\n".join(current_lines).strip()
            if block:
                elements.append(ContentElement(
                    element_type=current_type,
                    text=block,
                ))
            current_lines = []
            current_type = None

        for line in lines:
            stripped = line.strip()

            if stripped.startswith("|"):
                # Table line (content or separator like |---|---|)
                if current_type != ElementType.TABLE:
                    _flush()
                    current_type = ElementType.TABLE
                current_lines.append(line)

            elif stripped.startswith("#"):
                # Heading — each heading is its own element
                _flush()
                elements.append(ContentElement(
                    element_type=ElementType.TITLE,
                    text=stripped.lstrip("#").strip(),
                ))

            else:
                # Regular text
                if current_type != ElementType.TEXT:
                    _flush()
                    current_type = ElementType.TEXT
                current_lines.append(line)

        _flush()
        return elements

    def _extract_digital_pdf_with_vision(self, path: Path) -> ExtractedDocument:
        """
        Extract a digital/mixed PDF using unstructured + VLM enhancements.

        Uses unstructured for layout detection, then crops detected table
        regions and sends them to a VLM for accurate markdown extraction.
        PyMuPDF fallback catches missed embedded images.
        """
        from unstructured.partition.pdf import partition_pdf
        from app.document_processing.page_renderer import (
            render_page_to_image,
            crop_region_from_page,
        )

        logger.info("Digital PDF pipeline (unstructured + vision) for %s", path.name)

        elements = partition_pdf(
            str(path),
            strategy="hi_res",
            infer_table_structure=True,
            extract_image_block_to_payload=True,
        )

        # Group Table elements by page number for efficient rendering
        table_elements_by_page: dict[int, list] = {}
        for el in elements:
            if type(el).__name__ in ("Table", "TableChunk"):
                page_num = getattr(el.metadata, "page_number", 1) or 1
                table_elements_by_page.setdefault(page_num, []).append(el)

        # Render pages and crop table regions, then send to VLM
        temp_files: list[str] = []
        try:
            loop = asyncio.new_event_loop()
            try:
                for page_num, table_els in table_elements_by_page.items():
                    # Render this page once
                    page_img = render_page_to_image(str(path), page_num)
                    temp_files.append(page_img)

                    for el in table_els:
                        coords = getattr(el.metadata, "coordinates", None)
                        if not coords:
                            logger.warning(
                                "Table element on page %d has no coordinates, skipping vision",
                                page_num,
                            )
                            continue

                        points = coords.points
                        system = coords.system
                        page_w = system.width
                        page_h = system.height

                        # Crop the table region
                        crop_path = crop_region_from_page(
                            page_img, points, page_w, page_h
                        )
                        temp_files.append(crop_path)

                        # Send to VLM for extraction
                        try:
                            md_table = loop.run_until_complete(
                                self._vision_backend.extract_table(crop_path)
                            )
                            if md_table and md_table.strip() != "NO_TABLE_FOUND":
                                el._vision_table_text = md_table
                                logger.info(
                                    "Vision table extracted on page %d (%d chars)",
                                    page_num, len(md_table),
                                )
                            else:
                                logger.info(
                                    "Vision found no table on page %d, using unstructured fallback",
                                    page_num,
                                )
                        except Exception:
                            logger.exception(
                                "Vision table extraction failed on page %d, using fallback",
                                page_num,
                            )

                # Caption Image elements that have base64 payloads
                for el in elements:
                    if type(el).__name__ == "Image":
                        image_b64 = getattr(el.metadata, "image_base64", None)
                        if image_b64:
                            import base64
                            import tempfile as tf

                            img_bytes = base64.b64decode(image_b64)
                            tmp = tf.NamedTemporaryFile(
                                suffix=".png", prefix="img_element_", delete=False
                            )
                            tmp.write(img_bytes)
                            tmp.close()
                            temp_files.append(tmp.name)

                            try:
                                caption = loop.run_until_complete(
                                    self._vision_backend.caption_image(tmp.name)
                                )
                                el._vision_caption = caption
                            except Exception:
                                logger.exception("Vision captioning failed for image element")

                # --- PyMuPDF fallback: catch images unstructured missed ---
                from types import SimpleNamespace

                pymupdf_missed = self._collect_pymupdf_images(str(path))

                for pg_num, img_list in pymupdf_missed.items():
                    for tmp_path in img_list:
                        temp_files.append(tmp_path)
                        try:
                            caption = loop.run_until_complete(
                                self._vision_backend.caption_image(tmp_path)
                            )
                        except Exception:
                            logger.exception(
                                "Vision captioning failed for PyMuPDF image on page %d",
                                pg_num,
                            )
                            continue

                        # Create synthetic Image element compatible with _classify_element()
                        _SyntheticImage = type("Image", (), {})
                        synthetic_el = _SyntheticImage()
                        synthetic_el.metadata = SimpleNamespace(page_number=pg_num)
                        synthetic_el._vision_caption = caption
                        elements.append(synthetic_el)
            finally:
                loop.close()

            return self._elements_to_document(path, "pdf", elements)

        finally:
            for tmp_path in temp_files:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    # Maximum fraction of page area an image can cover before being
    # considered a full-page background/watermark (digital PDFs only).
    _PYMUPDF_PAGE_COVERAGE_THRESHOLD = 0.80

    def _collect_pymupdf_images(
        self,
        pdf_path: str,
    ) -> dict[int, list[str]]:
        """
        Find embedded images that unstructured may have missed using PyMuPDF.

        Only runs on digital/mixed PDFs (scanned PDFs use the VLM pipeline).
        Dedup is xref-based only — perceptual hashing removed for simplicity.
        Images covering >80% of the page area are skipped (backgrounds/watermarks).

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            Dict mapping 1-based page number to list of temp file paths.
        """
        import fitz
        from app.document_processing.page_renderer import extract_pymupdf_image_to_file

        result: dict[int, list[str]] = {}
        doc = fitz.open(pdf_path)

        try:
            for page_idx in range(len(doc)):
                page = doc[page_idx]
                page_num = page_idx + 1  # 1-based
                page_area = page.rect.width * page.rect.height
                missed: list[str] = []

                # --- Raster images via get_images() ---
                seen_xrefs: set[int] = set()
                for img_info in page.get_images(full=True):
                    xref = img_info[0]
                    if xref in seen_xrefs:
                        continue
                    seen_xrefs.add(xref)

                    # Size filter using PDF image metadata
                    try:
                        img_meta = doc.extract_image(xref)
                        if not img_meta or not img_meta.get("image"):
                            continue
                        w = img_meta.get("width", 0)
                        h = img_meta.get("height", 0)
                        if (
                            w < self._PYMUPDF_IMG_MIN_WIDTH
                            or h < self._PYMUPDF_IMG_MIN_HEIGHT
                            or w * h < self._PYMUPDF_IMG_MIN_AREA
                        ):
                            continue
                        # Skip full-page backgrounds/watermarks
                        if page_area > 0 and (w * h) / page_area > self._PYMUPDF_PAGE_COVERAGE_THRESHOLD:
                            continue
                    except Exception:
                        continue

                    tmp_path = extract_pymupdf_image_to_file(doc, xref)
                    if tmp_path:
                        missed.append(tmp_path)

                # --- Inline images via get_text("dict") ---
                try:
                    page_dict = page.get_text("dict")
                    for block in page_dict.get("blocks", []):
                        if block.get("type") != 1:  # type 1 = image block
                            continue

                        bbox = block.get("bbox")
                        if not bbox or len(bbox) < 4:
                            continue

                        w = bbox[2] - bbox[0]
                        h = bbox[3] - bbox[1]
                        if (
                            w < self._PYMUPDF_IMG_MIN_WIDTH
                            or h < self._PYMUPDF_IMG_MIN_HEIGHT
                            or w * h < self._PYMUPDF_IMG_MIN_AREA
                        ):
                            continue
                        # Skip full-page backgrounds/watermarks
                        if page_area > 0 and (w * h) / page_area > self._PYMUPDF_PAGE_COVERAGE_THRESHOLD:
                            continue

                        img_bytes = block.get("image")
                        if not img_bytes:
                            continue

                        try:
                            import io as _io
                            import tempfile as _tf
                            from PIL import Image as _PILImage

                            pil_img = _PILImage.open(_io.BytesIO(img_bytes))
                            if pil_img.mode not in ("RGB", "RGBA"):
                                pil_img = pil_img.convert("RGB")
                            tmp = _tf.NamedTemporaryFile(
                                suffix=".png", prefix="pymupdf_inline_", delete=False
                            )
                            pil_img.save(tmp.name, format="PNG")
                            tmp.close()
                            missed.append(tmp.name)
                        except Exception:
                            logger.exception(
                                "Failed to extract inline image on page %d", page_num
                            )
                except Exception:
                    logger.exception(
                        "Failed to parse inline images on page %d", page_num
                    )

                if missed:
                    result[page_num] = missed

            total = sum(len(v) for v in result.values())
            if total:
                logger.info(
                    "PyMuPDF found %d additional images across %d pages in %s",
                    total, len(result), pdf_path,
                )

        finally:
            doc.close()

        return result

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

    # Max rows per table chunk for Excel/CSV — keeps chunks manageable
    # for embedding and retrieval. ~50 rows × ~80 chars = ~4000 chars.
    XLSX_ROWS_PER_CHUNK = 50

    def _extract_xlsx(self, path: Path) -> ExtractedDocument:
        """
        Extract text from Excel spreadsheets using openpyxl.

        Large sheets are pre-chunked at extraction time: each chunk contains
        the header row + up to XLSX_ROWS_PER_CHUNK data rows. This avoids
        building a single giant markdown string and produces a reasonable
        number of chunks even for sheets with 50k+ rows.
        """
        import openpyxl

        wb = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
        pages = []
        total_rows = 0

        for sheet_idx, sheet_name in enumerate(wb.sheetnames, start=1):
            ws = wb[sheet_name]

            # Collect all non-empty rows, stringify cells
            all_rows: list[list[str]] = []
            for row in ws.iter_rows(values_only=True):
                if any(cell is not None for cell in row):
                    all_rows.append([str(cell) if cell is not None else "" for cell in row])

            if not all_rows:
                continue

            total_rows += len(all_rows)

            # First row is always the header
            header_row = all_rows[0]
            data_rows = all_rows[1:]

            # Pre-chunk: split data rows into groups of XLSX_ROWS_PER_CHUNK
            elements: list[ContentElement] = []

            if not data_rows:
                # Header-only sheet
                md = _rows_to_markdown([header_row])
                elements.append(ContentElement(
                    element_type=ElementType.TABLE,
                    text=md,
                    metadata={
                        "sheet_name": sheet_name,
                        "section_header": f"Sheet: {sheet_name}",
                        "row_range": "header only",
                    },
                ))
            else:
                for start in range(0, len(data_rows), self.XLSX_ROWS_PER_CHUNK):
                    batch = data_rows[start:start + self.XLSX_ROWS_PER_CHUNK]
                    # Always include header row so each chunk is self-contained
                    chunk_rows = [header_row] + batch
                    md = _rows_to_markdown(chunk_rows)

                    row_start = start + 2  # +2 because row 1 is header (1-indexed)
                    row_end = row_start + len(batch) - 1

                    elements.append(ContentElement(
                        element_type=ElementType.TABLE,
                        text=md,
                        metadata={
                            "sheet_name": sheet_name,
                            "section_header": f"Sheet: {sheet_name}",
                            "row_range": f"rows {row_start}-{row_end}",
                        },
                    ))

            # Build page text for full_text() (first chunk only for summary)
            page_text = f"### Sheet: {sheet_name} ({len(all_rows)} rows)\n\n{elements[0].text}"
            if len(elements) > 1:
                page_text += f"\n\n... and {len(elements) - 1} more table segments"

            pages.append(ExtractedPage(
                page_number=sheet_idx,
                text=page_text,
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
                "total_rows": total_rows,
            },
        )
        doc.total_text = doc.full_text()

        logger.info(
            f"Extracted {path.name}: {len(pages)} sheets, {total_rows} total rows, "
            f"{sum(len(p.elements) for p in pages)} table segments"
        )
        return doc

    def _extract_csv(self, path: Path) -> ExtractedDocument:
        """Extract text from CSV files, pre-chunked for large files."""
        text = path.read_text(encoding="utf-8-sig")
        reader = csv.reader(io.StringIO(text))
        all_rows = [row for row in reader if any(cell.strip() for cell in row)]

        if not all_rows:
            return ExtractedDocument(
                filename=path.name,
                file_type="csv",
                error="CSV file is empty",
            )

        header_row = all_rows[0]
        data_rows = all_rows[1:]

        elements: list[ContentElement] = []

        if not data_rows:
            md = _rows_to_markdown([header_row])
            elements.append(ContentElement(
                element_type=ElementType.TABLE,
                text=md,
                metadata={"row_count": 1},
            ))
        else:
            for start in range(0, len(data_rows), self.XLSX_ROWS_PER_CHUNK):
                batch = data_rows[start:start + self.XLSX_ROWS_PER_CHUNK]
                chunk_rows = [header_row] + batch
                md = _rows_to_markdown(chunk_rows)

                row_start = start + 2
                row_end = row_start + len(batch) - 1

                elements.append(ContentElement(
                    element_type=ElementType.TABLE,
                    text=md,
                    metadata={
                        "row_count": len(batch),
                        "row_range": f"rows {row_start}-{row_end}",
                    },
                ))

        page = ExtractedPage(
            page_number=1,
            text=elements[0].text,
            elements=elements,
            metadata={"source_file": path.name},
        )

        doc = ExtractedDocument(
            filename=path.name,
            file_type="csv",
            pages=[page],
            metadata={"source_path": str(path), "page_count": 1, "total_rows": len(all_rows)},
        )
        doc.total_text = doc.full_text()

        logger.info(
            f"Extracted {path.name}: {len(all_rows)} rows, "
            f"{len(elements)} table segments"
        )
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
            # Prefer vision-extracted table if available
            vision_text = getattr(element, "_vision_table_text", None)
            if vision_text:
                return ElementType.TABLE, vision_text

            # Try to get HTML representation for accurate table conversion
            html = getattr(element.metadata, "text_as_html", None)
            if html:
                md_table = _html_table_to_markdown(html)
                if md_table:
                    return ElementType.TABLE, md_table
            # Fallback: wrap in fenced code block to preserve spacing
            return ElementType.TABLE, f"```\n{str(element)}\n```"

        if element_class == "Image":
            # Use vision caption if available
            vision_caption = getattr(element, "_vision_caption", None)
            if vision_caption:
                return ElementType.TEXT, vision_caption

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
