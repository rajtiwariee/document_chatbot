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

    async def _call_endpoint(self, messages: list[dict]) -> str:
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
            "max_tokens": 1024,
            "temperature": 0.2,
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
