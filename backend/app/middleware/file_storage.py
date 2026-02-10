"""
Tenant-scoped file storage utilities.

Ensures uploaded files are stored in isolated directories per tenant,
preventing path traversal and cross-tenant file access.
"""
import os
import re
import uuid
from pathlib import Path

from app.config import get_settings

settings = get_settings()

# Base upload directory (from config or default)
UPLOAD_BASE = Path(getattr(settings, "upload_dir", "/app/uploads"))


def get_tenant_upload_dir(tenant_id: uuid.UUID) -> Path:
    """
    Get the upload directory for a specific tenant.
    Creates the directory if it doesn't exist.

    Structure: /app/uploads/{tenant_id}/
    """
    tenant_dir = UPLOAD_BASE / str(tenant_id)
    tenant_dir.mkdir(parents=True, exist_ok=True)
    return tenant_dir


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename to prevent path traversal and special characters.
    Returns a safe filename.
    """
    # Remove directory components
    filename = os.path.basename(filename)

    # Remove any characters that aren't alphanumeric, dots, dashes, or underscores
    safe_name = re.sub(r'[^\w\-.]', '_', filename)

    # Prevent hidden files and directory traversal
    if safe_name.startswith('.') or '..' in safe_name:
        safe_name = f"file_{safe_name.lstrip('.')}"

    # Ensure the filename isn't empty
    if not safe_name or safe_name == '_':
        safe_name = "unnamed_file"

    return safe_name


def generate_storage_path(
    tenant_id: uuid.UUID,
    original_filename: str,
) -> tuple[str, str]:
    """
    Generate a unique, tenant-isolated storage path for an uploaded file.

    Returns:
        (stored_filename, full_path_string)

    Example:
        ("a1b2c3d4_report.pdf", "/app/uploads/tenant-uuid/a1b2c3d4_report.pdf")
    """
    safe_name = sanitize_filename(original_filename)

    # Prefix with a short UUID to guarantee uniqueness
    unique_prefix = uuid.uuid4().hex[:8]
    stored_filename = f"{unique_prefix}_{safe_name}"

    tenant_dir = get_tenant_upload_dir(tenant_id)
    full_path = tenant_dir / stored_filename

    return stored_filename, str(full_path)


def validate_file_access(tenant_id: uuid.UUID, file_path: str) -> bool:
    """
    Validate that a file path belongs to the given tenant's directory.
    Prevents path traversal attacks.

    Returns True if the file path is within the tenant's upload directory.
    """
    tenant_dir = get_tenant_upload_dir(tenant_id)

    try:
        # Resolve the path to handle any .. tricks
        resolved_path = Path(file_path).resolve()
        resolved_tenant_dir = tenant_dir.resolve()

        # Check that the resolved path starts with the tenant's directory
        return str(resolved_path).startswith(str(resolved_tenant_dir))
    except (ValueError, OSError):
        return False
