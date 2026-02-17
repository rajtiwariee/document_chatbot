"""
Qwen-VL Vision backend — image captioning and VQA via Qwen3-VL
deployed on Vertex AI.

Qwen-VL on Vertex AI is served via vLLM and exposes an
OpenAI-compatible /v1/chat/completions endpoint. Images are
sent as base64 data URLs in the message content array.
"""
import base64
import logging
import mimetypes
from pathlib import Path

import httpx

from app.config import get_settings
from app.vision.base import VisionBackend

logger = logging.getLogger(__name__)

# Same captioning prompt as Gemini for consistency
CAPTION_PROMPT = """Describe this image in detail for a document search index.
Include all of the following that apply:
- Image type: chart, diagram, table, photo, screenshot, logo, etc.
- All visible text, labels, numbers, axis titles, and legends
- Data values, relationships, trends, and comparisons
- Layout, colors, and structural information
- Any context that would help someone find this image via text search
- If any tables are visible, reproduce them as markdown pipe-delimited tables (| col1 | col2 |) with exact values

Be thorough and factual. Do not speculate beyond what is visible."""


class QwenVLVision(VisionBackend):
    """Captioning and VQA using Qwen3-VL on Vertex AI (OpenAI-compatible API)."""

    def __init__(self):
        settings = get_settings()
        self.endpoint = settings.qwen_vl_endpoint.rstrip("/")
        self.api_key = settings.qwen_vl_api_key
        self.model = settings.vision_model  # e.g. "qwen3-vl-8b"
        self.timeout = 120  # seconds — vision inference can be slow

        if not self.endpoint:
            raise ValueError("QWEN_VL_ENDPOINT is required for Qwen-VL backend")

        logger.info(
            "QwenVLVision initialized (model: %s, endpoint: %s)",
            self.model, self.endpoint,
        )

    def _image_to_data_url(self, image_path: str) -> str:
        """Read an image file and encode as a base64 data URL."""
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        mime_type = mimetypes.guess_type(str(path))[0] or "image/png"
        b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
        return f"data:{mime_type};base64,{b64}"

    async def _call_endpoint(
        self,
        messages: list[dict],
        max_tokens: int = 1024,
        temperature: float = 0.2,
    ) -> str:
        """
        Send a chat completion request to the Qwen-VL Vertex AI endpoint.

        Uses the OpenAI-compatible /v1/chat/completions format.
        """
        url = f"{self.endpoint}/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()

        data = response.json()
        return data["choices"][0]["message"]["content"].strip()

    async def caption_image(self, image_path: str) -> str:
        """Generate a detailed text caption using Qwen3-VL."""
        logger.info("Captioning image with Qwen-VL: %s", image_path)

        data_url = self._image_to_data_url(image_path)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": CAPTION_PROMPT},
                ],
            }
        ]

        caption = await self._call_endpoint(messages)
        logger.info(
            "Qwen-VL caption generated (%d chars) for %s",
            len(caption), image_path,
        )
        return caption

    async def extract_table(self, image_path: str) -> str:
        """Extract table structure as markdown using Qwen3-VL."""
        logger.info("Extracting table with Qwen-VL: %s", image_path)

        data_url = self._image_to_data_url(image_path)
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

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        result = await self._call_endpoint(messages, max_tokens=5096, temperature=0.0)
        logger.info("Qwen-VL table extraction (%d chars) for %s", len(result), image_path)
        return result

    async def extract_page(self, image_path: str) -> str:
        """Extract all content from a full page image using Qwen3-VL."""
        logger.info("Extracting full page with Qwen-VL: %s", image_path)

        data_url = self._image_to_data_url(image_path)
        prompt = (
            "Extract ALL content from this document page image. "
            "Reproduce everything visible:\n\n"
            "- All text: reproduce paragraphs, headings, and lists exactly as written\n"
            "- Tables: convert to markdown pipe-delimited format (| col1 | col2 |) with exact values\n"
            "- Images/figures: describe them in [Image: ...] brackets\n"
            "- Preserve the reading order from top to bottom\n"
            "- Preserve headings hierarchy (use # for main headings, ## for subheadings)\n\n"
            "Be thorough and exact. Reproduce all text verbatim, do not summarize."
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        result = await self._call_endpoint(messages, max_tokens=8192, temperature=0.1)
        logger.info("Qwen-VL page extraction (%d chars) for %s", len(result), image_path)
        return result

    async def classify_image(self, image_path: str) -> str:
        """Classify image type using Qwen3-VL."""
        logger.info("Classifying image with Qwen-VL: %s", image_path)

        data_url = self._image_to_data_url(image_path)
        prompt = (
            "Classify this image into exactly ONE of these categories:\n"
            "table, chart, diagram, photo, screenshot, document, other\n\n"
            "Respond with a single word only."
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        result = await self._call_endpoint(messages, max_tokens=20, temperature=0.0)
        category = result.strip().lower().rstrip(".")
        valid = {"table", "chart", "diagram", "photo", "screenshot", "document", "other"}
        if category not in valid:
            logger.warning("Unexpected classification '%s', defaulting to 'other'", category)
            category = "other"
        logger.info("Qwen-VL classified %s as '%s'", image_path, category)
        return category

    async def visual_qa(self, image_path: str, question: str) -> str:
        """Answer a question about an image using Qwen3-VL."""
        logger.info("Qwen-VL VQA: '%s' on %s", question, image_path)

        data_url = self._image_to_data_url(image_path)
        prompt = (
            f"Look at this image and answer the following question accurately.\n\n"
            f"Question: {question}\n\n"
            f"Answer based only on what is visible in the image."
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        answer = await self._call_endpoint(messages)
        logger.info("Qwen-VL VQA answer (%d chars)", len(answer))
        return answer
