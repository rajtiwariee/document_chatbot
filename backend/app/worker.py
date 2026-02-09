import time
from celery import shared_task

@shared_task(name="process_document")
def process_document(document_id: str):
    """
    Placeholder task for processing a document.
    This will eventually handle text extraction, chunking, and embedding.
    """
    print(f"Processing document {document_id}...")
    time.sleep(5)  # Simulate processing time
    print(f"Document {document_id} processed successfully.")
    return {"status": "completed", "document_id": document_id}
