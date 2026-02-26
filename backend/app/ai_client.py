"""
Central AI client factory.

Single source of truth for creating Google AI clients. All modules should
import from here instead of initializing their own clients directly.

Vertex AI mode: GOOGLE_GENAI_USE_VERTEX=true + GOOGLE_CLOUD_PROJECT set → uses ADC (no API key)
API key mode (default): uses GOOGLE_API_KEY
"""
from app.config import get_settings


def get_genai_client():
    """Return a google.genai Client configured for Vertex AI or API key mode."""
    settings = get_settings()
    from google import genai
    if settings.use_vertex_ai:
        return genai.Client(
            vertexai=True,
            project=settings.google_cloud_project,
            location=settings.google_cloud_location,
        )
    return genai.Client(api_key=settings.google_api_key)


def get_langchain_llm(**kwargs):
    """Return a LangChain LLM: ChatVertexAI or ChatGoogleGenerativeAI."""
    settings = get_settings()
    if settings.use_vertex_ai:
        from langchain_google_vertexai import ChatVertexAI
        return ChatVertexAI(
            model=settings.gemini_model,
            project=settings.google_cloud_project,
            location=settings.google_cloud_location,
            **kwargs,
        )
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.google_api_key,
        **kwargs,
    )


def get_langchain_embeddings():
    """Return LangChain embeddings: VertexAIEmbeddings or GoogleGenerativeAIEmbeddings."""
    settings = get_settings()
    model_name = settings.embedding_model
    if settings.use_vertex_ai:
        from langchain_google_vertexai import VertexAIEmbeddings
        return VertexAIEmbeddings(
            model=model_name,
            project=settings.google_cloud_project,
            location=settings.google_cloud_location,
        )
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    if not model_name.startswith("models/"):
        model_name = f"models/{model_name}"
    return GoogleGenerativeAIEmbeddings(
        model=model_name,
        google_api_key=settings.google_api_key,
    )
