import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Manually set the key from .env (or simpler, just hardcode what we see to test)
# But better to load from env to match user's context
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
print(f"Testing API Key: {api_key[:10]}...{api_key[-5:]}")

from google import genai

print(f"Testing API Key: {api_key[:10]}...{api_key[-5:]}")

client = genai.Client(api_key=api_key)
print("Listing available models:")
try:        
    print("\nAttempting embedding with 'gemini-embedding-001'...")
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents="Hello world",
    )
    print(result.embeddings)
    print("SUCCESS: Native embedding call worked.")
except Exception as e:
    print(f"\nFAILURE: {e}")
