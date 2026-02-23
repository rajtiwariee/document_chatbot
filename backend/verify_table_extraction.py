"""
Script to verify Unstructured table extraction from images (charts, screenshots).

Usage:
    python verify_table_extraction.py /path/to/image.png

This script uses the 'hi_res' strategy to detect tables and extract their structure.
It prints out:
1. All detected elements (Text, Title, Table, Image)
2. For Tables: The HTML representation and text content
3. For Images: The element metadata (to confirm detection)
"""
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("verifier")

try:
    from unstructured.partition.image import partition_image
    from unstructured.partition.pdf import partition_pdf
except ImportError:
    logger.error("Error: 'unstructured' library not installed.")
    logger.error("pip install unstructured[pdf,image]")
    sys.exit(1)


def analyze_file(file_path: str):
    path = Path(file_path)
    if not path.exists():
        logger.error(f"File not found: {file_path}")
        return

    logger.info(f"Analyzing {file_path}...")

    try:
        if path.suffix.lower() == ".pdf":
            elements = partition_pdf(
                str(path),
                strategy="hi_res",
                infer_table_structure=True,
                extract_image_block_to_payload=True,
                extract_image_block_types=["Image", "Table"],
            )
        else:
            elements = partition_image(
                str(path),
                strategy="hi_res",
                infer_table_structure=True,
                extract_image_block_to_payload=True,
                extract_image_block_types=["Image", "Table"],
            )
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        return

    logger.info(f"Found {len(elements)} elements.")

    # Build markdown output
    md_lines = [
        f"# Extraction Results: `{path.name}`\n",
        f"**Source:** `{file_path}`  ",
        f"**Elements found:** {len(elements)}\n",
        "---\n",
    ]

    for i, el in enumerate(elements):
        el_type = type(el).__name__
        text = str(el).strip()

        md_lines.append(f"## Element {i+1}: {el_type}\n")

        if el_type == "Table":
            html = getattr(el.metadata, "text_as_html", None)
            if html:
                md_lines.append("### Table (HTML)\n")
                md_lines.append(f"{html}\n")
            if text:
                md_lines.append("### Table (Plain Text)\n")
                md_lines.append(f"```\n{text}\n```\n")
        elif el_type == "Title":
            md_lines.append(f"**{text}**\n")
        else:
            if text:
                md_lines.append(f"{text}\n")

        has_base64 = hasattr(el.metadata, "image_base64") and el.metadata.image_base64
        md_lines.append(f"*Image base64:* {'Present' if has_base64 else 'Not present'}\n")
        md_lines.append("---\n")

    # Write markdown file next to the input file
    output_path = path.with_suffix(".md")
    output_path.write_text("\n".join(md_lines), encoding="utf-8")
    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python verify_table_extraction.py <path_to_image_or_pdf>")
        sys.exit(1)

    analyze_file(sys.argv[1])
