"""
Gemini Vision backend — image captioning and VQA via Google Gemini API.

Uses the google.genai SDK (same as the rest of the codebase) to send
images to Gemini for captioning and visual question answering.
"""
import base64
import logging
import mimetypes
from pathlib import Path

from google.genai import types

from app.config import get_settings
from app.ai_client import get_genai_client
from app.vision.base import VisionBackend

logger = logging.getLogger(__name__)

# Captioning prompt — designed for maximum retrieval quality
CAPTION_PROMPT = """Describe this image in detail for a document search index.
Include all of the following that apply:
- Image type: chart, diagram, table, photo, screenshot, logo, etc.
- All visible text, labels, numbers, axis titles, and legends
- Data values, relationships, trends, and comparisons
- Layout, colors, and structural information
- Any context that would help someone find this image via text search
- If any tables are visible, reproduce them as markdown pipe-delimited tables (| col1 | col2 |) with exact values

Be thorough and factual. Do not speculate beyond what is visible."""


class GeminiVision(VisionBackend):
    """Captioning and VQA using Google Gemini Vision API."""

    def __init__(self):
        settings = get_settings()
        self.client = get_genai_client()
        self.model = settings.vision_model
        logger.info("GeminiVision initialized (model: %s)", self.model)

    def _load_image_part(self, image_path: str) -> types.Part:
        """Load an image file and return a genai Part for the API."""
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        mime_type = mimetypes.guess_type(str(path))[0] or "image/png"
        image_bytes = path.read_bytes()

        return types.Part.from_bytes(data=image_bytes, mime_type=mime_type)

    async def caption_image(self, image_path: str) -> str:
        """Generate a detailed text caption using Gemini Vision."""
        logger.info("Captioning image with Gemini: %s", image_path)

        image_part = self._load_image_part(image_path)
        response = self.client.models.generate_content(
            model=self.model,
            contents=[CAPTION_PROMPT, image_part],
            config=types.GenerateContentConfig(
                temperature=0.2,  # Low temp for factual descriptions
                max_output_tokens=1024,
            ),
        )

        caption = response.text.strip()
        logger.info(
            "Gemini caption generated (%d chars) for %s",
            len(caption), image_path,
        )
        return caption

    async def extract_table(self, image_path: str) -> str:
        """Extract table structure as markdown using Gemini Vision."""
        logger.info("Extracting table with Gemini: %s", image_path)

        image_part = self._load_image_part(image_path)
        prompt = (
            "Extract the table from this image into a markdown pipe-delimited table.\n\n"
            "Rules:\n"
            "- Reproduce EVERY row and column exactly as shown\n"
            "- Use | to separate columns and --- for the header separator\n"
            "- Preserve all numbers, text, and formatting precisely\n"
            "- If cells are merged, repeat the value in each cell\n"
            "- If no table is found, respond with exactly: NO_TABLE_FOUND\n\n"
            "Output only the markdown table, nothing else."
        )

        response = self.client.models.generate_content(
            model=self.model,
            contents=[prompt, image_part],
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=4096,
            ),
        )

        result = response.text.strip()
        logger.info("Gemini table extraction (%d chars) for %s", len(result), image_path)
        return result

    async def extract_page(self, image_path: str) -> str:
        """Extract all content from a full page image using Gemini Vision."""
        logger.info("Extracting full page with Gemini: %s", image_path)

        image_part = self._load_image_part(image_path)
        prompt = (
            "You are a precise document content extraction system. Extract EVERY piece of "
            "content from this document page image. Missing even one row or section is a failure.\n\n"
            "## Output Format\n\n"
            "- **Headings**: Use # for main headings, ## for subheadings, exactly as they appear.\n"
            "- **Paragraphs**: Reproduce text exactly as written.\n"
            "- **Tables**: Convert ALL tabular data to markdown pipe-delimited tables.\n"
            "  Example format:\n"
            "  | Column A | Column B | Column C |\n"
            "  | --- | --- | --- |\n"
            "  | data 1 | data 2 | data 3 |\n"
            "  | data 4 | data 5 | data 6 |\n\n"
            "## Visual Element Rules\n\n"
            "- **Data charts** (pie, bar, line, scatter): Write a tag like "
            "[Chart: Title or description], then extract ALL data points as a markdown "
            "pipe table — labels, values, percentages, units. Example:\n"
            "  [Chart: Revenue Breakdown by Region]\n"
            "  | Region | Revenue | Percentage |\n"
            "  | --- | --- | --- |\n"
            "  | North America | $4.5M | 45% |\n"
            "  | Europe | $3.0M | 30% |\n\n"
            "- **Flowcharts / process diagrams**: Write [Diagram: Title], then describe "
            "each step as a numbered list with arrows showing flow:\n"
            "  [Diagram: Order Processing Flow]\n"
            "  1. Customer places order\n"
            "  2. -> Payment validation\n"
            "  3. -> Inventory check -> If out of stock: notify customer\n"
            "  4. -> Ship order\n\n"
            "- **Hierarchy / org charts**: Write [Diagram: Title], then use indented "
            "nested lists to show the structure.\n"
            "- **Other images** (logos, photos, decorative): Describe briefly in "
            "[Image: ...] brackets.\n"
            "- For ALL visual types: extract EVERY visible label, number, and text element. "
            "Never skip data points.\n\n"
            "## Table Extraction Rules\n\n"
            "- Identify ALL tabular data — including price lists, comparison data, or "
            "aligned columns — even if they lack visible grid lines.\n"
            "- First, identify the column headers. Then place every data row into the correct columns.\n"
            "- If the page has MULTIPLE tables, give each one its own heading (## Table Title) "
            "before the pipe-delimited output.\n"
            "- Data separated by dots, dashes, or whitespace alignment is tabular — "
            "extract it as a pipe table, not as raw text.\n\n"
            "## Critical Rules\n\n"
            "- Start at the TOP and work to the BOTTOM. Do NOT stop until the entire page is done.\n"
            "- Do NOT summarize, abbreviate, or skip repetitive rows. Every row matters.\n"
            "- Do NOT say 'continued' or '...' — output the actual content.\n"
            "- Reproduce all numbers, dates, and values exactly as shown.\n"
            "- If a table has many rows (10, 20, 50+), you MUST include ALL of them.\n\n"
            "Begin extraction now."
        )

        response = self.client.models.generate_content(
            model=self.model,
            contents=[prompt, image_part],
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=30000,
            ),
        )

        # Check finish reason for debugging
        finish_reason = None
        if response.candidates:
            finish_reason = response.candidates[0].finish_reason
            if str(finish_reason) == "MAX_TOKENS":
                logger.warning(
                    "Gemini extract_page hit MAX_TOKENS for %s — output likely truncated",
                    image_path,
                )
            elif str(finish_reason) not in ("STOP", "FinishReason.STOP", "0", "None"):
                logger.warning(
                    "Gemini extract_page unusual finish_reason=%s for %s",
                    finish_reason, image_path,
                )

        result = response.text.strip()
        logger.info(
            "Gemini page extraction (%d chars, finish=%s) for %s",
            len(result), finish_reason, image_path,
        )
        return result

    async def classify_image(self, image_path: str) -> str:
        """Classify image type using Gemini Vision."""
        logger.info("Classifying image with Gemini: %s", image_path)

        image_part = self._load_image_part(image_path)
        prompt = (
            "Classify this image into exactly ONE of these categories:\n"
            "table, chart, diagram, photo, screenshot, document, other\n\n"
            "Respond with a single word only."
        )

        response = self.client.models.generate_content(
            model=self.model,
            contents=[prompt, image_part],
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=20,
            ),
        )

        category = response.text.strip().lower().rstrip(".")
        valid = {"table", "chart", "diagram", "photo", "screenshot", "document", "other"}
        if category not in valid:
            logger.warning("Unexpected classification '%s', defaulting to 'other'", category)
            category = "other"
        logger.info("Gemini classified %s as '%s'", image_path, category)
        return category

    async def visual_qa(self, image_path: str, question: str) -> str:
        """Answer a question about an image using Gemini Vision."""
        logger.info("Gemini VQA: '%s' on %s", question, image_path)

        image_part = self._load_image_part(image_path)
        prompt = (
            f"Look at this image and answer the following question accurately.\n\n"
            f"Question: {question}\n\n"
            f"Answer based only on what is visible in the image."
        )

        response = self.client.models.generate_content(
            model=self.model,
            contents=[prompt, image_part],
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=512,
            ),
        )

        answer = response.text.strip()
        logger.info("Gemini VQA answer (%d chars)", len(answer))
        return answer
