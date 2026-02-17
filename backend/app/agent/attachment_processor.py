"""
Chat attachment processor.

Handles ephemeral file attachments in chat messages:
- Images → base64 data URLs for multimodal LLM input
- Documents (PDF/DOCX) → extracted text via DocumentExtractor
- Spreadsheets (CSV/XLSX) → pandas DataFrames for query tool
"""
import base64
import io
import logging
import os
import shutil
import tempfile
import uuid
from dataclasses import dataclass, field

import pandas as pd
from fastapi import UploadFile

from app.document_processing.extractor import DocumentExtractor

logger = logging.getLogger(__name__)

# Module-level registry for active DataFrames (scoped to request lifetime)
_active_dataframes: dict[str, dict] = {}

MIME_TO_CATEGORY = {
    "image/png": "image",
    "image/jpeg": "image",
    "image/gif": "image",
    "image/webp": "image",
    "application/pdf": "document",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "document",
    "text/csv": "document",  # overridden below for spreadsheet
    "application/csv": "document",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "document",
}

IMAGE_MIMES = {"image/png", "image/jpeg", "image/gif", "image/webp"}
DOCUMENT_MIMES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}
SPREADSHEET_MIMES = {
    "text/csv",
    "application/csv",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}
ALLOWED_MIMES = IMAGE_MIMES | DOCUMENT_MIMES | SPREADSHEET_MIMES

MAX_DOC_TEXT_CHARS = 15_000


def _classify_mime(mime: str, filename: str) -> str:
    """Classify a file into image/document/spreadsheet category."""
    if mime in IMAGE_MIMES:
        return "image"
    if mime in SPREADSHEET_MIMES:
        return "spreadsheet"
    # Also check by extension for CSVs that come with wrong MIME
    ext = os.path.splitext(filename)[1].lower()
    if ext in (".csv", ".xlsx"):
        return "spreadsheet"
    if mime in DOCUMENT_MIMES:
        return "document"
    return "document"


@dataclass
class ProcessedAttachment:
    """A processed chat attachment."""
    attachment_id: str
    filename: str
    category: str  # "image", "document", "spreadsheet"
    image_data_url: str | None = None
    extracted_text: str | None = None
    spreadsheet_summary: str | None = None


@dataclass
class AttachmentContext:
    """Wraps processed attachments with cleanup capability."""
    attachments: list[ProcessedAttachment] = field(default_factory=list)
    temp_dir: str | None = None
    _dataframe_ids: list[str] = field(default_factory=list)

    def cleanup(self):
        """Remove temp files and clear DataFrame registry entries."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        for att_id in self._dataframe_ids:
            _active_dataframes.pop(att_id, None)


def _generate_spreadsheet_summary(df: pd.DataFrame, filename: str) -> str:
    """Generate a concise summary of a DataFrame for LLM context."""
    lines = [f"Spreadsheet: {filename}"]
    lines.append(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    lines.append(f"Columns: {', '.join(df.columns.tolist())}")
    lines.append(f"Dtypes:\n{df.dtypes.to_string()}")
    lines.append(f"First 3 rows:\n{df.head(3).to_string()}")
    return "\n".join(lines)


async def process_attachments(files: list[UploadFile]) -> AttachmentContext:
    """
    Process uploaded files into attachment context.

    For each file:
    - Images: encode as base64 data URL
    - Documents: extract text via DocumentExtractor
    - Spreadsheets: load into DataFrame, store in registry, generate summary
    """
    ctx = AttachmentContext()
    if not files:
        return ctx

    ctx.temp_dir = tempfile.mkdtemp(prefix="chat_attachments_")

    for upload_file in files:
        att_id = str(uuid.uuid4())
        filename = upload_file.filename or f"file_{att_id}"
        mime = upload_file.content_type or "application/octet-stream"
        category = _classify_mime(mime, filename)

        # Save to temp dir
        temp_path = os.path.join(ctx.temp_dir, f"{att_id}_{filename}")
        content = await upload_file.read()
        with open(temp_path, "wb") as f:
            f.write(content)

        attachment = ProcessedAttachment(
            attachment_id=att_id,
            filename=filename,
            category=category,
        )

        try:
            if category == "image":
                encoded = base64.b64encode(content).decode("utf-8")
                attachment.image_data_url = f"data:{mime};base64,{encoded}"
                logger.info(
                    f"Processed image attachment: {filename}, mime={mime}, "
                    f"raw_size={len(content)}bytes, data_url_len={len(attachment.image_data_url)}"
                )

            elif category == "document":
                extractor = DocumentExtractor()
                ext = os.path.splitext(filename)[1].lower().lstrip(".")
                file_type = ext if ext else "pdf"
                result = extractor.extract(temp_path, file_type)
                text = ""
                if result.pages:
                    text = "\n\n".join(
                        p.text for p in result.pages if p.text
                    )
                if len(text) > MAX_DOC_TEXT_CHARS:
                    text = text[:MAX_DOC_TEXT_CHARS] + "\n\n[Content truncated...]"
                attachment.extracted_text = text

            elif category == "spreadsheet":
                ext = os.path.splitext(filename)[1].lower()
                if ext == ".csv":
                    df = pd.read_csv(io.BytesIO(content))
                else:
                    df = pd.read_excel(io.BytesIO(content))

                _active_dataframes[att_id] = {
                    "df": df,
                    "filename": filename,
                }
                ctx._dataframe_ids.append(att_id)
                attachment.spreadsheet_summary = _generate_spreadsheet_summary(df, filename)

        except Exception as e:
            logger.error(f"Error processing attachment {filename}: {e}")
            attachment.extracted_text = f"[Error processing file: {e}]"

        ctx.attachments.append(attachment)

    return ctx


def get_dataframe(attachment_id: str) -> pd.DataFrame | None:
    """Get an active DataFrame by attachment ID."""
    entry = _active_dataframes.get(attachment_id)
    return entry["df"] if entry else None


def get_dataframe_info(attachment_id: str) -> dict | None:
    """Get DataFrame info (df + filename) by attachment ID."""
    return _active_dataframes.get(attachment_id)
