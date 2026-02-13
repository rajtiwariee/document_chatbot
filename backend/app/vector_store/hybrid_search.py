"""
Hybrid search combining vector similarity (Qdrant) with BM25 keyword matching.

Fusion via Reciprocal Rank Fusion (RRF) for optimal recall on both
semantic and keyword queries (e.g. part numbers like "BRK-45821").
"""
import logging
import re
from collections import defaultdict

from rank_bm25 import BM25Okapi

from app.document_processing.embeddings import EmbeddingGenerator
from app.vector_store.store import TenantVectorStore, SearchResult

logger = logging.getLogger(__name__)

# RRF constant (standard value from the original paper)
RRF_K = 60


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer for BM25."""
    return re.findall(r"\w+", text.lower())


class HybridSearcher:
    """
    Combines vector search and BM25 keyword search via Reciprocal Rank Fusion.

    BM25 index is built lazily per tenant by scrolling Qdrant payloads.
    Cache is invalidated after document add/delete.
    """

    def __init__(
        self,
        vector_store: TenantVectorStore | None = None,
        embedding_generator: EmbeddingGenerator | None = None,
        vector_weight: float = 0.6,
    ):
        self.vector_store = vector_store or TenantVectorStore()
        self.embedding_generator = embedding_generator or EmbeddingGenerator()
        self.vector_weight = vector_weight
        self.bm25_weight = 1.0 - vector_weight

        # Cache: tenant_id -> (BM25Okapi, list of payload dicts)
        self._bm25_cache: dict[str, tuple[BM25Okapi, list[dict]]] = {}

    def _build_bm25_index(self, tenant_id: str) -> tuple[BM25Okapi, list[dict]]:
        """Build BM25 index by scrolling all points in a tenant's Qdrant collection."""
        collection_name = self.vector_store._collection_name(tenant_id)

        try:
            self.vector_store.client.get_collection(collection_name)
        except Exception:
            logger.warning(f"Collection {collection_name} not found for BM25 build")
            return BM25Okapi([[""]]), []

        all_payloads: list[dict] = []
        all_tokens: list[list[str]] = []

        offset = None
        batch_size = 256
        while True:
            results, next_offset = self.vector_store.client.scroll(
                collection_name=collection_name,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            if not results:
                break

            for point in results:
                payload = point.payload or {}
                text = payload.get("text", "")
                all_payloads.append(payload)
                all_tokens.append(_tokenize(text))

            offset = next_offset
            if offset is None:
                break

        if not all_tokens:
            return BM25Okapi([[""]]), []

        bm25 = BM25Okapi(all_tokens)
        logger.info(f"Built BM25 index for tenant {tenant_id}: {len(all_payloads)} documents")
        return bm25, all_payloads

    def _get_bm25(self, tenant_id: str) -> tuple[BM25Okapi, list[dict]]:
        """Get or build BM25 index for a tenant (cached)."""
        if tenant_id not in self._bm25_cache:
            self._bm25_cache[tenant_id] = self._build_bm25_index(tenant_id)
        return self._bm25_cache[tenant_id]

    def invalidate_cache(self, tenant_id: str) -> None:
        """Invalidate BM25 cache for a tenant (call after document add/delete)."""
        self._bm25_cache.pop(tenant_id, None)
        logger.info(f"Invalidated BM25 cache for tenant {tenant_id}")

    def search(
        self,
        tenant_id: str,
        query: str,
        top_k: int = 5,
        document_id: str | None = None,
    ) -> list[SearchResult]:
        """
        Hybrid search: vector + BM25, fused with RRF.

        Args:
            tenant_id: Tenant scope
            query: Natural language query
            top_k: Number of results to return
            document_id: Optional filter to a specific document

        Returns:
            Fused list of SearchResult ordered by combined RRF score
        """
        fetch_k = top_k * 2

        # --- Vector search ---
        query_embedding = self.embedding_generator.embed_query(query)
        vector_results = self.vector_store.search(
            tenant_id=tenant_id,
            query_embedding=query_embedding,
            top_k=fetch_k,
            document_id=document_id,
        )

        # --- BM25 search ---
        bm25, payloads = self._get_bm25(tenant_id)
        bm25_results: list[SearchResult] = []

        if payloads:
            query_tokens = _tokenize(query)
            scores = bm25.get_scores(query_tokens)

            # Build (index, score) pairs and sort
            scored = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

            for idx, score in scored[:fetch_k]:
                if score <= 0:
                    break
                payload = payloads[idx]

                # Filter by document_id if specified
                if document_id and payload.get("document_id") != document_id:
                    continue

                bm25_results.append(SearchResult(
                    chunk_text=payload.get("text", ""),
                    document_id=payload.get("document_id", ""),
                    page_number=payload.get("page_number"),
                    score=float(score),
                    metadata={
                        "source_file": payload.get("source_file", ""),
                        "file_type": payload.get("file_type", ""),
                        "chunk_index": payload.get("chunk_index", 0),
                        "element_type": payload.get("element_type", "text"),
                        "section_header": payload.get("section_header", ""),
                        "sheet_name": payload.get("sheet_name", ""),
                    },
                ))

        # --- Reciprocal Rank Fusion ---
        # Key: chunk_text hash -> (rrf_score, SearchResult)
        rrf_scores: dict[str, float] = defaultdict(float)
        result_map: dict[str, SearchResult] = {}

        for rank, result in enumerate(vector_results):
            key = f"{result.document_id}:{result.metadata.get('chunk_index', 0)}"
            rrf_scores[key] += self.vector_weight * (1.0 / (RRF_K + rank + 1))
            result_map[key] = result

        for rank, result in enumerate(bm25_results):
            key = f"{result.document_id}:{result.metadata.get('chunk_index', 0)}"
            rrf_scores[key] += self.bm25_weight * (1.0 / (RRF_K + rank + 1))
            if key not in result_map:
                result_map[key] = result

        # Sort by RRF score
        sorted_keys = sorted(rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True)

        fused_results = []
        for key in sorted_keys[:top_k]:
            result = result_map[key]
            # Replace score with RRF score for downstream consumers
            fused_results.append(SearchResult(
                chunk_text=result.chunk_text,
                document_id=result.document_id,
                page_number=result.page_number,
                score=rrf_scores[key],
                metadata=result.metadata,
            ))

        return fused_results
