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
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")  # Default to a Gemini model
    embedding_model: str = "gemini-embedding-001"
    embedding_dimensions: int = 3072  # gemini-embedding-001
    enable_reranking: bool = True

    # Vertex AI (for GCP VM with Application Default Credentials)
    google_genai_use_vertex: bool = False
    google_cloud_project: str = ""
    google_cloud_location: str = "us-central1"

    @property
    def use_vertex_ai(self) -> bool:
        return self.google_genai_use_vertex and bool(self.google_cloud_project)

    # Vision Backend (for multimodal RAG image captioning & VQA)
    vision_backend: str = os.getenv("VISION_BACKEND", "gemini")  # "gemini" or "qwen_vl"
    vision_model: str = os.getenv("VISION_MODEL", "gemini-3-flash-preview")  # Gemini model or "qwen3-vl-8b"
    qwen_vl_endpoint: str = ""  # Vertex AI endpoint URL for Qwen-VL
    qwen_vl_api_key: str = ""  # API key/token for Qwen-VL endpoint
    image_storage_dir: str = "./uploads/images"  # Persistent image storage
    enable_vision_table_extraction: bool = False  # Use VLM for PDF table extraction

    # File Upload & Storage
    upload_dir: str = "./uploads"
    max_file_size_mb: int = 100

    # Chat Attachments
    chat_attachment_max_size_mb: int = 10
    chat_attachment_max_count: int = 3
    storage_backend: str = "local"    # "local" or "gcs"
    gcs_bucket_name: str = ""

    # Vertex AI Vector Search (for production)
    vertex_vector_index_id: str = os.getenv("VERTEX_VECTOR_INDEX_ID", "")
    vertex_vector_endpoint_id: str = os.getenv("VERTEX_VECTOR_ENDPOINT_ID", "")
    vertex_vector_location: str = os.getenv("VERTEX_VECTOR_LOCATION", "us-central1")

    @property
    def use_vertex_vector_search(self) -> bool:
        """Check if Vertex AI Vector Search should be used instead of Qdrant."""
        return bool(self.vertex_vector_index_id) and bool(self.vertex_vector_endpoint_id)
    
    
    class Config:
        env_file = [".env", "../.env"]
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
