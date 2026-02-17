"""
PDF page rendering and region cropping for vision-enhanced table extraction.

Uses PyMuPDF (fitz) for page rendering and Pillow for region cropping.
Both libraries are already project dependencies.
"""
import logging
import tempfile
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image

logger = logging.getLogger(__name__)

# Minimum characters of extractable text for a page to be considered "digital"
_DIGITAL_TEXT_THRESHOLD = 50


def detect_pdf_type(pdf_path: str) -> tuple[str, dict[int, str]]:
    """
    Classify a PDF as "scanned", "digital", or "mixed".

    Inspects every page:
    - If extractable text > threshold → page is digital.
    - If little/no text but raster images present → page is scanned.

    Returns:
        Tuple of (overall_type, page_types) where:
        - overall_type: "scanned", "digital", or "mixed"
        - page_types: dict mapping 1-based page number to "scanned" or "digital"
    """
    doc = fitz.open(pdf_path)
    try:
        digital_count = 0
        scanned_count = 0
        page_types: dict[int, str] = {}

        for page_idx, page in enumerate(doc):
            page_num = page_idx + 1
            text = page.get_text("text").strip()
            if len(text) > _DIGITAL_TEXT_THRESHOLD:
                digital_count += 1
                page_types[page_num] = "digital"
            elif page.get_images():
                scanned_count += 1
                page_types[page_num] = "scanned"
            else:
                # Blank or near-blank page with no images — treat as digital
                digital_count += 1
                page_types[page_num] = "digital"

        total = digital_count + scanned_count
        if total == 0:
            return "digital", page_types
        if scanned_count == total:
            return "scanned", page_types
        if digital_count == total:
            return "digital", page_types
        return "mixed", page_types
    finally:
        doc.close()


def extract_pymupdf_image_to_file(doc: fitz.Document, xref: int) -> str | None:
    """
    Extract an image by xref from an open fitz.Document and save as temp PNG.

    Handles CMYK → RGB conversion via PIL.

    Returns:
        Path to a temporary PNG file, or None on failure.
        Caller is responsible for cleanup.
    """
    try:
        img_info = doc.extract_image(xref)
        if not img_info or not img_info.get("image"):
            return None

        img_bytes = img_info["image"]
        import io
        pil_img = Image.open(io.BytesIO(img_bytes))

        # Convert CMYK (or other non-RGB modes) to RGB for consistent PNG output
        if pil_img.mode not in ("RGB", "RGBA"):
            pil_img = pil_img.convert("RGB")

        tmp = tempfile.NamedTemporaryFile(
            suffix=".png", prefix="pymupdf_img_", delete=False
        )
        pil_img.save(tmp.name, format="PNG")
        tmp.close()
        return tmp.name
    except Exception:
        logger.exception("Failed to extract PyMuPDF image xref=%d", xref)
        return None


def render_page_to_image(pdf_path: str, page_number: int, dpi: int = 200) -> str:
    """
    Render a single PDF page as a PNG image.

    Args:
        pdf_path: Path to the PDF file.
        page_number: 1-based page number.
        dpi: Resolution for rendering (default 200).

    Returns:
        Path to a temporary PNG file. Caller is responsible for cleanup.
    """
    doc = fitz.open(pdf_path)
    try:
        page = doc[page_number - 1]  # fitz uses 0-based indexing
        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        pixmap = page.get_pixmap(matrix=matrix)

        tmp = tempfile.NamedTemporaryFile(
            suffix=".png", prefix=f"page_{page_number}_", delete=False
        )
        pixmap.save(tmp.name)
        tmp.close()

        logger.debug(
            "Rendered page %d of %s → %s (%dx%d)",
            page_number, pdf_path, tmp.name, pixmap.width, pixmap.height,
        )
        return tmp.name
    finally:
        doc.close()


def crop_region_from_page(
    page_image_path: str,
    bbox_points: list[tuple[float, float]],
    page_width: float,
    page_height: float,
    padding: int = 10,
) -> str:
    """
    Crop a region from a rendered page image using bounding box coordinates.

    Unstructured provides coordinates as 4 corner points in PDF point space
    via element.metadata.coordinates.points, with the coordinate system
    dimensions in element.metadata.coordinates.system.width/height.

    Args:
        page_image_path: Path to the rendered page PNG.
        bbox_points: List of 4 (x, y) tuples from unstructured coordinates.
        page_width: Width of the coordinate system (PDF points).
        page_height: Height of the coordinate system (PDF points).
        padding: Extra pixels around the crop (default 10).

    Returns:
        Path to a temporary cropped PNG file. Caller is responsible for cleanup.
    """
    img = Image.open(page_image_path)
    img_w, img_h = img.size

    # Scale factors from PDF point space to pixel space
    scale_x = img_w / page_width
    scale_y = img_h / page_height

    # Get bounding rectangle from corner points
    xs = [p[0] for p in bbox_points]
    ys = [p[1] for p in bbox_points]

    left = max(0, int(min(xs) * scale_x) - padding)
    top = max(0, int(min(ys) * scale_y) - padding)
    right = min(img_w, int(max(xs) * scale_x) + padding)
    bottom = min(img_h, int(max(ys) * scale_y) + padding)

    cropped = img.crop((left, top, right, bottom))

    tmp = tempfile.NamedTemporaryFile(
        suffix=".png", prefix="table_crop_", delete=False
    )
    cropped.save(tmp.name, format="PNG")
    tmp.close()

    logger.debug(
        "Cropped region (%d,%d)-(%d,%d) from %s → %s (%dx%d)",
        left, top, right, bottom, page_image_path, tmp.name,
        cropped.width, cropped.height,
    )
    return tmp.name
