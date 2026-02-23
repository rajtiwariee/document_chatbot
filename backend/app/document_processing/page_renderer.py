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


def analyze_pages(pdf_path: str) -> list[dict]:
    """
    Analyze each page to decide VLM vs text extraction.

    For each page:
    - If it has meaningful images, tables, OR very little extractable text → VLM.
    - Otherwise → text extraction with PyMuPDF.

    Returns:
        List of dicts (one per page):
        {"page_num": 1, "has_images": True, "has_tables": False,
         "text_len": 234, "method": "vlm"|"text"}
    """
    doc = fitz.open(pdf_path)
    results = []
    try:
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            text = page.get_text("text").strip()

            # Check for meaningful images (not tiny icons/decorations)
            has_images = False
            for img_info in page.get_images(full=True):
                try:
                    img_meta = doc.extract_image(img_info[0])
                    w = img_meta.get("width", 0)
                    h = img_meta.get("height", 0)
                    if w >= 50 and h >= 50 and w * h >= 10_000:
                        has_images = True
                        break
                except Exception:
                    continue

            # Check for tables using PyMuPDF's built-in table detection
            has_tables = False
            try:
                tables = page.find_tables()
                has_tables = len(tables.tables) > 0
            except Exception:
                pass

            # Decision: images, tables, or very little text → VLM
            method = "vlm" if has_images or has_tables or len(text) < _DIGITAL_TEXT_THRESHOLD else "text"

            results.append({
                "page_num": page_idx + 1,
                "has_images": has_images,
                "has_tables": has_tables,
                "text_len": len(text),
                "method": method,
            })
    finally:
        doc.close()
    return results


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
