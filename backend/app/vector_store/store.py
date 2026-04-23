"""
Multi-tenant vector store wrapper supporting Qdrant and Vertex AI Vector Search.

Each tenant gets an isolated namespace/collection structure.
- Qdrant: Uses a separate collection per tenant (`tenant_{tenant_id}`).
- Vertex AI Vector Search: Uses the same index but isolates via `restricts` (metadata).
"""
import logging
from typing import List, Dict, Any, Optional

from app.config import get_settings
from app.document_processing.chunker import Chunk

settings = get_settings()
logger = logging.getLogger(__name__)


class SearchResult:
    """A single search result from vector store."""
    def __init__(self, chunk_text: str, document_id: str, page_number: Optional[int], score: float, metadata: Dict[str, Any] = None):
        self.chunk_text = chunk_text
        self.document_id = document_id
        self.page_number = page_number
        self.score = score
        self.metadata = metadata or {}


class TenantVectorStore:
    """
    Unified Vector Store interface for Qdrant and Vertex AI Vector Search.
    
    If `use_vertex_vector_search` is True in Config, it connects to Google Cloud.
    Otherwise, it defaults to the local Qdrant container.
    """

    def __init__(self):
        self.vector_size = settings.embedding_dimensions
        self.use_vertex = settings.use_vertex_vector_search

        if self.use_vertex:
            logger.info("Initializing Vertex AI Vector Search...")
            from google.cloud import aiplatform
            aiplatform.init(project=settings.google_cloud_project, location=settings.vertex_vector_location)
            
            self.index_endpoint = aiplatform.MatchingEngineIndexEndpoint(
                index_endpoint_name=settings.vertex_vector_endpoint_id
            )
            self.index = aiplatform.MatchingEngineIndex(
                index_name=settings.vertex_vector_index_id
            )
        else:
            logger.info("Initializing Qdrant Vector Store...")
            from qdrant_client import QdrantClient
            self.client = QdrantClient(
                host=settings.qdrant_host,
                port=settings.qdrant_port,
                timeout=10,
            )

    # =========================================================================
    # QDRANT SPECIFIC METHODS
    # =========================================================================

    def _qdrant_collection_name(self, tenant_id: str) -> str:
        """Generate collection name for a tenant."""
        return f"tenant_{tenant_id}"

    def _qdrant_get_or_create_collection(self, tenant_id: str) -> str:
        """Ensure a tenant-specific collection exists in Qdrant."""
        from qdrant_client.models import Distance, VectorParams, PayloadSchemaType

        collection_name = self._qdrant_collection_name(tenant_id)

        try:
            collections = self.client.get_collections().collections
            existing_names = [c.name for c in collections]

            if collection_name not in existing_names:
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE,
                    ),
                )
                logger.info(f"Created Qdrant collection: {collection_name}")

            # Ensure payload index exists for efficiency
            collection_info = self.client.get_collection(collection_name)
            existing_indexes = set(collection_info.payload_schema.keys()) if collection_info.payload_schema else set()

            if "document_id" not in existing_indexes:
                self.client.create_payload_index(
                    collection_name=collection_name,
                    field_name="document_id",
                    field_schema=PayloadSchemaType.KEYWORD,
                )
        except Exception as e:
            logger.error(f"Error checking Qdrant collection {collection_name}: {e}")
            raise

        return collection_name
        
    def _delete_qdrant_tenant_collection(self, tenant_id: str) -> None:
        """Delete an entire tenant's collection in Qdrant."""
        collection_name = self._qdrant_collection_name(tenant_id)
        try:
            self.client.delete_collection(collection_name)
            logger.info(f"Deleted collection: {collection_name}")
        except Exception as e:
            logger.error(f"Error deleting collection {collection_name}: {e}")

    # =========================================================================
    # UNIFIED METHODS
    # =========================================================================

    def add_chunks(
        self,
        tenant_id: str,
        chunks: List[Chunk],
        embeddings: List[List[float]],
    ) -> int:
        """
        Add document chunks with their embeddings to the vector store.
        """
        if not chunks or not embeddings:
            return 0

        if self.use_vertex:
            return self._vertex_add_chunks(tenant_id, chunks, embeddings)
        else:
            return self._qdrant_add_chunks(tenant_id, chunks, embeddings)

    def search(
        self,
        tenant_id: str,
        query_embedding: List[float],
        top_k: int = 5,
        document_id: str | None = None,
    ) -> List[SearchResult]:
        """
        Search within a tenant's documents.
        """
        if self.use_vertex:
            return self._vertex_search(tenant_id, query_embedding, top_k, document_id)
        else:
            return self._qdrant_search(tenant_id, query_embedding, top_k, document_id)

    def delete_document_vectors(self, tenant_id: str, document_id: str) -> None:
        """
        Delete all vectors for a specific document within a tenant.
        """
        if self.use_vertex:
            self._vertex_delete_document_vectors(tenant_id, document_id)
        else:
            self._qdrant_delete_document_vectors(tenant_id, document_id)

    # -------------------------------------------------------------------------
    # VERTEX AI IMPLEMENTATION
    # -------------------------------------------------------------------------

    def _vertex_add_chunks(
        self,
        tenant_id: str,
        chunks: List[Chunk],
        embeddings: List[List[float]],
    ) -> int:
        """Store vectors in Vertex AI Vector Search."""
        from google.cloud.aiplatform.matching_engine.matching_engine_index_endpoint import Namespace
        import json
        
        datapoints = []
        for chunk, embedding in zip(chunks, embeddings):
            # Categorical filters (Restricts)
            # We enforce tenant_id isolation here.
            restricts = [
                Namespace(name="tenant_id", allow_tokens=[tenant_id]),
                Namespace(name="document_id", allow_tokens=[chunk.document_id])
            ]
            
            # Additional structured metadata (optional, but helps with debug if needed)
            if chunk.metadata.get("file_type"):
                restricts.append(Namespace(name="file_type", allow_tokens=[chunk.metadata["file_type"]]))
            
            # Vertex AI Vector Search allows string payloads via 'crowding_tag' (small)
            # or in newer versions via 'embedding_metadata' (up to 2KB).
            # The python SDK requires building a dict matching the REST API for advanced fields,
            # or we can use the `IndexDatapoint` class.
            
            # `google.cloud.aiplatform.matching_engine.matching_engine_index_endpoint.IndexDatapoint` 
            # does not strictly have an `embedding_metadata` kwarg in older SDKs.
            # However, we can use the `restricts` trick or string payload if it fits. 
            # Alternatively, since we are using BigQuery later or need the text *right now* for search:
            # We carefully serialize the chunk text into a metadata dict to return it on match.
            # *UPDATE*: Since `chunk.text` can be long, we might hit Vertex 2KB payload limits if we use 
            # certain fields. We'll store a shortened version if needed, or rely on Postgres/GCS.
            # For this exact implementation we'll serialize to JSON and put into `restricts` as a hack 
            # if `embedding_metadata` isn't available, but standard approach is returning document_id and fetching text.
            
            from google.cloud.aiplatform.matching_engine.matching_engine_index_endpoint import IndexDatapoint
            
            # To preserve backwards compatibility with Qdrant, we MUST return `chunk_text`.
            # We will use the `crowding_tag` for string metadata if it fits, else we fall back 
            # to storing the full text in the ID string (format: UUID|||JSON_PAYLOAD).
            # This is a common pattern for Vector DBs without dedicated large text payload fields.
            
            payload = {
                "text": chunk.text,
                "document_id": chunk.document_id,
                "page_number": chunk.page_number,
                "chunk_index": chunk.chunk_index,
                "source_file": chunk.metadata.get("source_file", ""),
                "file_type": chunk.metadata.get("file_type", ""),
                "element_type": chunk.metadata.get("element_type", "text"),
                "section_header": chunk.metadata.get("section_header", ""),
                "sheet_name": chunk.metadata.get("sheet_name", ""),
            }
            
            # Format: uuid:::json_string. Vertex AI supports IDs up to 500 characters.
            # If our payload is larger, it will truncate.
            # Proper fix for production: Fetch the chunk text from a DB (like Postgres or BigQuery)
            # instead of putting the whole text directly into the Vector DB.
            # Since `store.py` assumes the vector store holds the text, we'll pack it in the ID or crowding_tag.
            
            # Clean text (remove newlines) to keep size down
            safe_text = chunk.text.replace("\n", " ")
            if len(safe_text) > 200:
                safe_text = safe_text[:197] + "..."
                
            payload_str = json.dumps(payload)
            # If payload is extremely large, we limit the text.
            if len(payload_str) > 400:
                payload["text"] = safe_text
                payload_str = json.dumps(payload)
                
            datapoint_id = f"{chunk.id}:::{payload_str}"
            
            # Datapoint IDs must be <= 500 characters in Vertex AI.
            if len(datapoint_id) > 500:
                # Fallback: Just store essential routing info in ID if text is too huge.
                min_payload = {"d_id": chunk.document_id, "idx": chunk.chunk_index}
                datapoint_id = f"{chunk.id}:::{json.dumps(min_payload)}"
            
            dp = IndexDatapoint(
                datapoint_id=datapoint_id,
                feature_vector=embedding,
                restricts=restricts,
            )
            datapoints.append(dp)
            
        # Insert using streaming update
        if datapoints:
            self.index_endpoint.mutate_fixed_datapoints(
                index=self.index.name,
                datapoints=datapoints,
            )

        logger.info(f"Upserted {len(datapoints)} vectors to Vertex AI Vector Search")
        return len(datapoints)

    def _vertex_search(
        self,
        tenant_id: str,
        query_embedding: List[float],
        top_k: int = 5,
        document_id: str | None = None,
    ) -> List[SearchResult]:
        from google.cloud.aiplatform.matching_engine.matching_engine_index_endpoint import Namespace
        import json

        # Mandatory tenant-level isolation filter
        restricts = [
            Namespace(name="tenant_id", allow_tokens=[tenant_id])
        ]
        
        # Optional document-level filter
        if document_id:
            restricts.append(Namespace(name="document_id", allow_tokens=[document_id]))

        # Perform the nearest neighbor search
        response = self.index_endpoint.find_neighbors(
            deployed_index_id=settings.vertex_vector_index_id, # Usually deployed ID matches
            queries=[query_embedding],
            num_neighbors=top_k,
            filter=restricts,
        )

        results = []
        if response and len(response) > 0:
            for neighbor in response[0]:
                chunk_id = neighbor.id
                metadata = {}
                score = neighbor.distance # Vertex returns distance, needs conversion if cosine
                
                # Unpack the ID format: UUID:::JSON
                if ":::" in chunk_id:
                    parts = chunk_id.split(":::", 1)
                    try:
                        metadata = json.loads(parts[1])
                    except json.JSONDecodeError:
                        pass
                
                # Note: If text was truncated during insert due to Vertex AI 500-char ID limits, 
                # we return the truncated text here. 
                # A true production system fetching text from Postgres/BigQuery would be called 
                # at this step using `parts[0]` (the chunk UUID).
                
                results.append(SearchResult(
                    chunk_text=metadata.get("text", chunk_id),
                    document_id=metadata.get("document_id", ""),
                    page_number=metadata.get("page_number"),
                    score=score,
                    metadata=metadata
                ))
                
        return results

    def _vertex_delete_document_vectors(self, tenant_id: str, document_id: str) -> None:
        from google.cloud.aiplatform.matching_engine.matching_engine_index_endpoint import Namespace

        # Vertex AI Vector Search does not support direct "delete by filter" easily.
        # However, we can use the `find_neighbors` with the document_id restrict to fetch 
        # the list of datapoint IDs, and then issue a delete command for those specific IDs.
        
        restricts = [
            Namespace(name="tenant_id", allow_tokens=[tenant_id]),
            Namespace(name="document_id", allow_tokens=[document_id])
        ]
        
        # We need to find all vectors for this document. 
        # Since we use Cosine distance, ANY vector will match to find nearest neighbors. 
        # A dummy vector works because we just need to retrieve the IDs filtered by restricts.
        dummy_vector = [0.0] * self.vector_size
        dummy_vector[0] = 1.0
        
        try:
            response = self.index_endpoint.find_neighbors(
                deployed_index_id=settings.vertex_vector_index_id,
                queries=[dummy_vector],
                num_neighbors=1000, # Assume max 1000 chunks per document for deletion
                filter=restricts,
            )
            
            if response and len(response) > 0:
                datapoint_ids_to_delete = [neighbor.id for neighbor in response[0]]
                if datapoint_ids_to_delete:
                    self.index_endpoint.mutate_fixed_datapoints(
                        index=self.index.name,
                        datapoint_ids=datapoint_ids_to_delete
                    )
                    logger.info(f"Deleted {len(datapoint_ids_to_delete)} vectors for document {document_id}")
        except Exception as e:
            logger.error(f"Error deleting vectors for document {document_id} from Vertex AI: {e}")


    # -------------------------------------------------------------------------
    # QDRANT IMPLEMENTATION (Legacy)
    # -------------------------------------------------------------------------

    def _qdrant_add_chunks(
        self,
        tenant_id: str,
        chunks: List[Chunk],
        embeddings: List[List[float]],
    ) -> int:
        from qdrant_client.models import PointStruct

        collection_name = self._qdrant_get_or_create_collection(tenant_id)
        points = [
            PointStruct(
                id=chunk.id,
                vector=embedding,
                payload={
                    "text": chunk.text,
                    "document_id": chunk.document_id,
                    "tenant_id": chunk.tenant_id,
                    "page_number": chunk.page_number,
                    "chunk_index": chunk.chunk_index,
                    "source_file": chunk.metadata.get("source_file", ""),
                    "file_type": chunk.metadata.get("file_type", ""),
                    "element_type": chunk.metadata.get("element_type", "text"),
                    "section_header": chunk.metadata.get("section_header", ""),
                    "sheet_name": chunk.metadata.get("sheet_name", ""),
                    "content_type": chunk.metadata.get("content_type", "text"),
                    "original_image_path": chunk.metadata.get("original_image_path", ""),
                },
            )
            for chunk, embedding in zip(chunks, embeddings)
        ]

        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self.client.upsert(
                collection_name=collection_name,
                points=batch,
            )

        logger.info(f"Upserted {len(points)} vectors to Qdrant {collection_name}")
        return len(points)

    def _qdrant_search(
        self,
        tenant_id: str,
        query_embedding: List[float],
        top_k: int = 5,
        document_id: str | None = None,
    ) -> List[SearchResult]:
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        collection_name = self._qdrant_collection_name(tenant_id)
        try:
            self.client.get_collection(collection_name)
        except Exception:
            return []

        query_filter = None
        if document_id:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id),
                    )
                ]
            )

        response = self.client.query_points(
            collection_name=collection_name,
            query=query_embedding,
            query_filter=query_filter,
            limit=top_k,
        )

        return [
            SearchResult(
                chunk_text=hit.payload.get("text", ""),
                document_id=hit.payload.get("document_id", ""),
                page_number=hit.payload.get("page_number"),
                score=hit.score,
                metadata={
                    "source_file": hit.payload.get("source_file", ""),
                    "file_type": hit.payload.get("file_type", ""),
                    "chunk_index": hit.payload.get("chunk_index", 0),
                    "element_type": hit.payload.get("element_type", "text"),
                    "section_header": hit.payload.get("section_header", ""),
                    "sheet_name": hit.payload.get("sheet_name", ""),
                    "content_type": hit.payload.get("content_type", "text"),
                    "original_image_path": hit.payload.get("original_image_path", ""),
                },
            )
            for hit in response.points
        ]

    def _qdrant_delete_document_vectors(self, tenant_id: str, document_id: str) -> None:
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        collection_name = self._qdrant_collection_name(tenant_id)
        try:
            self.client.delete(
                collection_name=collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=MatchValue(value=document_id),
                        )
                    ]
                ),
            )
        except Exception as e:
            logger.error(f"Error deleting vectors in Qdrant: {e}")

