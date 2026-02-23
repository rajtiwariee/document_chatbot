"""
Centralized logging configuration.

Call `setup_logging()` once at app startup (in main.py) and once
in the Celery worker signal handler so forked workers inherit it.

Features:
- Structured JSON logs for production (machine-parseable)
- Human-readable colored logs for development
- Rotating file output (10MB per file, keeps 5 backups)
- Separate error log file
- Request context (tenant_id, user_id) via log extras
- Celery worker-safe: re-configures after fork
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

        # Context fields
        for attr in ("tenant_id", "user_id", "document_id"):
            val = getattr(record, attr, None)
            if val is not None:
                log_entry[attr] = val

        if record.exc_info and record.exc_info[1]:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry)


class ReadableFormatter(logging.Formatter):
    """Human-readable C-style formatter for development.

    Format:
        HH:MM:SS.mmm  [LEVEL]  module.path:line  func()  message
    Example:
        13:54:36.480  [INFO ]  app.api.auth:140  login()  User logged in
    """

    COLORS = {
        "DEBUG":    "\033[36m",      # Cyan
        "INFO":     "\033[32m",      # Green
        "WARNING":  "\033[33m",      # Yellow
        "ERROR":    "\033[31m",      # Red
        "CRITICAL": "\033[1;31m",    # Bold Red
    }
    RESET = "\033[0m"
    DIM   = "\033[90m"

    def format(self, record: logging.LogRecord) -> str:
        color  = self.COLORS.get(record.levelname, self.RESET)
        ts     = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        level  = f"{color}[{record.levelname:<5}]{self.RESET}"
        source = f"{self.DIM}{record.name}:{record.lineno}{self.RESET}"
        func   = f"{self.DIM}{record.funcName}(){self.RESET}"
        msg    = record.getMessage()

        extras = []
        for attr in ("tenant_id", "user_id", "document_id"):
            val = getattr(record, attr, None)
            if val is not None:
                short = str(val)[:8] if len(str(val)) > 8 else str(val)
                extras.append(f"{attr}={short}")
        extra_str = f"  {self.DIM}[{', '.join(extras)}]{self.RESET}" if extras else ""

        line = f"{ts}  {level}  {source}  {func}  {msg}{extra_str}"

        if record.exc_info and record.exc_info[1]:
            line += "\n" + self.formatException(record.exc_info)

        return line


def setup_logging() -> None:
    """
    Configure logging for the entire application.

    Safe to call multiple times (clears existing handlers first).
    Called by:
    - main.py lifespan (FastAPI)
    - celery_app.py worker_process_init signal (each forked worker)
    """
    os.makedirs(LOG_DIR, exist_ok=True)

    log_level = logging.DEBUG if settings.debug else logging.INFO

    # ── Root logger ───────────────────────────────────────────────
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.handlers.clear()

    # ── Console handler ───────────────────────────────────────────
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)

    if settings.debug:
        console_handler.setFormatter(ReadableFormatter())
    else:
        console_handler.setFormatter(JSONFormatter())

    root_logger.addHandler(console_handler)

    # ── Rotating file handler (all logs) ──────────────────────────
    file_handler = logging.handlers.RotatingFileHandler(
        filename=os.path.join(LOG_DIR, "app.log"),
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(JSONFormatter())
    root_logger.addHandler(file_handler)

    # ── Error-only file handler ───────────────────────────────────
    error_handler = logging.handlers.RotatingFileHandler(
        filename=os.path.join(LOG_DIR, "error.log"),
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(JSONFormatter())
    root_logger.addHandler(error_handler)

    # ── App loggers: ensure they propagate to root ────────────────
    # Explicitly set our app loggers to DEBUG so nothing is missed
    for app_logger_name in [
        "app",
        "app.agent",
        "app.agent.tools",
        "app.agent.graph",
        "app.document_processing",
        "app.document_processing.extractor",
        "app.document_processing.chunker",
        "app.document_processing.embeddings",
        "app.document_processing.page_renderer",
        "app.vector_store",
        "app.vector_store.store",
        "app.vector_store.hybrid_search",
        "app.vector_store.reranker",
        "app.vision",
        "app.vision.base",
        "app.vision.gemini_vision",
        "app.vision.qwen_vl_vision",
        "app.api",
        "app.api.chat",
        "app.api.documents",
        "app.api.auth",
        "app.worker",
        "app.middleware",
        "app.websockets",
    ]:
        app_log = logging.getLogger(app_logger_name)
        app_log.setLevel(logging.DEBUG)
        # Don't add handlers — let them propagate to root
        app_log.propagate = True

    # ── Quiet noisy third-party loggers ───────────────────────────
    for noisy_logger in [
        "uvicorn",
        "uvicorn.error",
        "uvicorn.access",
        "sqlalchemy.engine",
        "celery",
        "celery.worker",
        "celery.app.trace",
        "celery.redirected",
        "httpcore",
        "httpx",
        "urllib3",
        "unstructured",
        "unstructured.trace",
        "PIL",
        "pdfminer",
        "pytesseract",
        "google",
        "google.auth",
        "google.api_core",
        "google_genai",
        "grpc",
        "qdrant_client",
        "rank_bm25",
        "langchain",
        "langchain_core",
        "langsmith",
        "multipart",
        "asyncio",
    ]:
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    # Let uvicorn.error through at INFO (startup/shutdown messages)
    logging.getLogger("uvicorn.error").setLevel(logging.INFO)
    # Let celery task-level logs through
    logging.getLogger("celery").setLevel(logging.INFO)

    logger = logging.getLogger("app")
    logger.info("Logging configured: level=%s, log_dir=%s, debug=%s", log_level, LOG_DIR, settings.debug)
