"""
Application configuration using Pydantic Settings.
"""
from functools import lru_cache
from pydantic_settings import BaseSettings
import os

class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Application
    app_name: str = "Document Chatbot"
    debug: bool = False
    log_dir: str = "./logs"
    
    # Database
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/chatbot"
    
    # Redis
    redis_url: str = "redis://localhost:6379"
    
    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    
    # JWT
    secret_key: str = "your-secret-key-change-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # Google Gemini
    google_api_key: str = os.getenv("GOOGLE_API_KEY")
    gemini_model: str = "gemini-3-flash-preview"
    embedding_model: str = "gemini-embedding-001"
    embedding_dimensions: int = 3072  # gemini-embedding-001
    enable_reranking: bool = True

    # Vision Backend (for multimodal RAG image captioning & VQA)
    vision_backend: str = "gemini"  # "gemini" or "qwen_vl"
    vision_model: str = "gemini-2.0-flash"  # Gemini model or "qwen3-vl-8b"
    qwen_vl_endpoint: str = ""  # Vertex AI endpoint URL for Qwen-VL
    qwen_vl_api_key: str = ""  # API key/token for Qwen-VL endpoint
    image_storage_dir: str = "./uploads/images"  # Persistent image storage

    # File Upload & Storage
    upload_dir: str = "./uploads"
    max_file_size_mb: int = 100

    # Chat Attachments
    chat_attachment_max_size_mb: int = 10
    chat_attachment_max_count: int = 3
    storage_backend: str = "local"    # "local" or "gcs"
    gcs_bucket_name: str = ""
    
    class Config:
        env_file = [".env", "../.env"]
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
