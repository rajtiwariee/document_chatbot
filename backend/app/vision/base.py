"""
Abstract vision backend and factory function.

Provides a strategy pattern for swapping between Gemini and
Qwen-VL vision backends via the VISION_BACKEND environment variable.
"""
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class VisionBackend(ABC):
    """Abstract base class for vision backends (captioning + VQA)."""

    @abstractmethod
    async def caption_image(self, image_path: str) -> str:
        """
        Generate a detailed text description of an image.

        The caption should include: image type (chart/diagram/photo/etc.),
        all visible text/labels/numbers, data relationships and trends,
        layout and structural information.

        Args:
            image_path: Absolute path to the image file on disk.

        Returns:
            A detailed text caption suitable for embedding and search.
        """
        ...

    @abstractmethod
    async def visual_qa(self, image_path: str, question: str) -> str:
        """
        Answer a question about an image (Visual Question Answering).

        Args:
            image_path: Absolute path to the image file on disk.
            question: The user's natural language question about the image.

        Returns:
            A text answer derived from the image content.
        """
        ...


def get_vision_backend() -> VisionBackend:
    """
    Factory: return the configured vision backend based on settings.

    Reads ``settings.vision_backend`` and returns the matching
    implementation. Raises ValueError for unknown backends.
    """
    from app.config import get_settings

    settings = get_settings()
    backend = settings.vision_backend.lower().strip()

    if backend == "gemini":
        from app.vision.gemini_vision import GeminiVision
        logger.info("Using Gemini vision backend (model: %s)", settings.vision_model)
        return GeminiVision()

    if backend == "qwen_vl":
        from app.vision.qwen_vl_vision import QwenVLVision
        if not settings.qwen_vl_endpoint:
            raise ValueError(
                "QWEN_VL_ENDPOINT must be set when VISION_BACKEND=qwen_vl"
            )
        logger.info(
            "Using Qwen-VL vision backend (model: %s, endpoint: %s)",
            settings.vision_model,
            settings.qwen_vl_endpoint,
        )
        return QwenVLVision()

    raise ValueError(
        f"Unknown vision backend: '{backend}'. Must be 'gemini' or 'qwen_vl'."
    )
