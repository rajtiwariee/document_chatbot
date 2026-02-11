"""
Switchable file storage backends.

- Local: Files stored on disk at /app/uploads/{tenant_id}/
- GCS: Files stored in Google Cloud Storage at gs://{bucket}/{tenant_id}/

Switch via config: STORAGE_BACKEND=local or STORAGE_BACKEND=gcs
"""
import os
import re
import uuid
import logging
from abc import ABC, abstractmethod
from pathlib import Path

from app.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)


def sanitize_filename(filename: str) -> str:
    """Sanitize a filename to prevent path traversal and special characters."""
    filename = os.path.basename(filename)
    safe_name = re.sub(r'[^\w\-.]', '_', filename)

    if safe_name.startswith('.') or '..' in safe_name:
        safe_name = f"file_{safe_name.lstrip('.')}"

    if not safe_name or safe_name == '_':
        safe_name = "unnamed_file"

    return safe_name


def _generate_stored_filename(original_filename: str) -> str:
    """Generate a unique filename with a short UUID prefix."""
    safe_name = sanitize_filename(original_filename)
    unique_prefix = uuid.uuid4().hex[:8]
    return f"{unique_prefix}_{safe_name}"


# ---------------------------------------------------------------------------
# Abstract Base
# ---------------------------------------------------------------------------
class StorageBackend(ABC):
    """Abstract storage backend. All methods are tenant-scoped."""

    @abstractmethod
    def save(self, tenant_id: uuid.UUID, filename: str, content: bytes) -> str:
        """Save file content. Returns the storage path/key."""

    @abstractmethod
    def read(self, tenant_id: uuid.UUID, path: str) -> bytes:
        """Read file content by path/key."""

    @abstractmethod
    def delete(self, tenant_id: uuid.UUID, path: str) -> None:
        """Delete a file by path/key."""

    @abstractmethod
    def validate_access(self, tenant_id: uuid.UUID, path: str) -> bool:
        """Check if the path belongs to the given tenant."""


# ---------------------------------------------------------------------------
# Local Disk Storage
# ---------------------------------------------------------------------------
class LocalStorage(StorageBackend):
    """Store files on the local filesystem."""

    def __init__(self, base_dir: str = "./uploads"):
        self.base_dir = Path(base_dir)

    def _tenant_dir(self, tenant_id: uuid.UUID) -> Path:
        tenant_dir = self.base_dir / str(tenant_id)
        tenant_dir.mkdir(parents=True, exist_ok=True)
        return tenant_dir

    def save(self, tenant_id: uuid.UUID, filename: str, content: bytes) -> str:
        stored_name = _generate_stored_filename(filename)
        full_path = self._tenant_dir(tenant_id) / stored_name
        with open(full_path, "wb") as f:
            f.write(content)
        logger.info(f"Saved locally: {full_path}")
        return str(full_path)

    def read(self, tenant_id: uuid.UUID, path: str) -> bytes:
        if not self.validate_access(tenant_id, path):
            raise PermissionError("Access denied")
        with open(path, "rb") as f:
            return f.read()

    def delete(self, tenant_id: uuid.UUID, path: str) -> None:
        if self.validate_access(tenant_id, path) and os.path.exists(path):
            os.remove(path)
            logger.info(f"Deleted locally: {path}")

    def validate_access(self, tenant_id: uuid.UUID, path: str) -> bool:
        try:
            resolved = Path(path).resolve()
            tenant_dir = self._tenant_dir(tenant_id).resolve()
            return str(resolved).startswith(str(tenant_dir))
        except (ValueError, OSError):
            return False


# ---------------------------------------------------------------------------
# Google Cloud Storage
# ---------------------------------------------------------------------------
class GCSStorage(StorageBackend):
    """Store files in Google Cloud Storage."""

    def __init__(self, bucket_name: str):
        from google.cloud import storage as gcs

        self.client = gcs.Client()
        self.bucket = self.client.bucket(bucket_name)

    def _blob_path(self, tenant_id: uuid.UUID, filename: str) -> str:
        """Generate blob path: {tenant_id}/{filename}"""
        return f"{tenant_id}/{filename}"

    def save(self, tenant_id: uuid.UUID, filename: str, content: bytes) -> str:
        stored_name = _generate_stored_filename(filename)
        blob_path = self._blob_path(tenant_id, stored_name)
        blob = self.bucket.blob(blob_path)
        blob.upload_from_string(content)
        logger.info(f"Saved to GCS: gs://{self.bucket.name}/{blob_path}")
        return blob_path

    def read(self, tenant_id: uuid.UUID, path: str) -> bytes:
        if not self.validate_access(tenant_id, path):
            raise PermissionError("Access denied")
        blob = self.bucket.blob(path)
        return blob.download_as_bytes()

    def delete(self, tenant_id: uuid.UUID, path: str) -> None:
        if self.validate_access(tenant_id, path):
            blob = self.bucket.blob(path)
            if blob.exists():
                blob.delete()
                logger.info(f"Deleted from GCS: {path}")

    def validate_access(self, tenant_id: uuid.UUID, path: str) -> bool:
        return path.startswith(f"{tenant_id}/")


# ---------------------------------------------------------------------------
# Factory: pick the backend based on config
# ---------------------------------------------------------------------------
def get_storage() -> StorageBackend:
    """Return the configured storage backend."""
    backend = getattr(settings, "storage_backend", "local")

    if backend == "gcs":
        bucket = getattr(settings, "gcs_bucket_name", "")
        if not bucket:
            raise ValueError("GCS_BUCKET_NAME must be set when STORAGE_BACKEND=gcs")
        return GCSStorage(bucket_name=bucket)

    return LocalStorage(base_dir=settings.upload_dir)
