"""
Centralized logging configuration.

Call `setup_logging()` once at app startup (in main.py).
All modules using `logging.getLogger(__name__)` will automatically
pick up this configuration.

Features:
- Structured JSON logs for production (machine-parseable)
- Human-readable colored logs for development
- Rotating file output (10MB per file, keeps 5 backups)
- Separate error log file
- Request context (tenant_id, user_id) via log filters
"""
import logging
import logging.handlers
import json
import sys
import os
from datetime import datetime, timezone

from app.config import get_settings

settings = get_settings()

LOG_DIR = settings.log_dir


class JSONFormatter(logging.Formatter):
    """Structured JSON log formatter for production."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        if hasattr(record, "tenant_id"):
            log_entry["tenant_id"] = record.tenant_id
        if hasattr(record, "user_id"):
            log_entry["user_id"] = record.user_id
        if hasattr(record, "document_id"):
            log_entry["document_id"] = record.document_id

        if record.exc_info and record.exc_info[1]:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry)


class ReadableFormatter(logging.Formatter):
    """Human-readable formatter for development."""

    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[1;31m",# Bold Red
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.RESET)
        timestamp = datetime.now().strftime("%H:%M:%S")

        prefix = f"{color}{timestamp} {record.levelname:8s}{self.RESET}"
        location = f"\033[90m{record.name}:{record.lineno}\033[0m"
        message = record.getMessage()

        extras = []
        if hasattr(record, "tenant_id"):
            extras.append(f"tenant={record.tenant_id}")
        if hasattr(record, "document_id"):
            extras.append(f"doc={record.document_id}")
        extra_str = f" \033[90m[{', '.join(extras)}]\033[0m" if extras else ""

        formatted = f"{prefix} {location} {message}{extra_str}"

        if record.exc_info and record.exc_info[1]:
            formatted += "\n" + self.formatException(record.exc_info)

        return formatted


def setup_logging() -> None:
    """
    Configure logging for the entire application.
    Call once at startup.
    """
    os.makedirs(LOG_DIR, exist_ok=True)

    log_level = logging.DEBUG if settings.debug else logging.INFO

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Clear any existing handlers (root + uvicorn loggers)
    root_logger.handlers.clear()
    for _name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logging.getLogger(_name).handlers.clear()

    # --- Console handler ---
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)

    if settings.debug:
        console_handler.setFormatter(ReadableFormatter())
    else:
        console_handler.setFormatter(JSONFormatter())

    root_logger.addHandler(console_handler)

    # --- Rotating file handler (all logs) ---
    file_handler = logging.handlers.RotatingFileHandler(
        filename=os.path.join(LOG_DIR, "app.log"),
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(JSONFormatter())
    root_logger.addHandler(file_handler)

    # --- Error-only file handler ---
    error_handler = logging.handlers.RotatingFileHandler(
        filename=os.path.join(LOG_DIR, "error.log"),
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(JSONFormatter())
    root_logger.addHandler(error_handler)

    # --- Quiet noisy third-party loggers ---
    logging.getLogger("uvicorn.access").setLevel(logging.INFO)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    logging.getLogger("celery").setLevel(logging.INFO)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    logging.info("Logging configured: level=%s, dir=%s", log_level, LOG_DIR)
