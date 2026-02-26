"""
LLM-based reranker using Gemini Flash as a cross-encoder.

Takes hybrid search results (top 10-15), scores each passage against
the query, and returns the top results by relevance.
"""
import json
import logging

from google.genai import types

from app.config import get_settings
from app.ai_client import get_genai_client
from app.vector_store.store import SearchResult

settings = get_settings()
logger = logging.getLogger(__name__)

RERANK_PROMPT = """You are a relevance scoring system. Given a query and a list of document passages, score each passage from 0 to 10 based on how relevant it is to answering the query.

Query: {query}

Passages:
{passages}

Return ONLY a JSON array of integers representing the scores for each passage in order. Example: [8, 3, 10, 1, 5]
Do not include any explanation or text outside the JSON array."""


class GeminiReranker:
    """Rerank search results using Gemini Flash as a cross-encoder."""

    def __init__(self, top_n: int = 5):
        self.top_n = top_n
        self.client = get_genai_client()

    def rerank(self, query: str, results: list[SearchResult]) -> list[SearchResult]:
        """
        Rerank search results by LLM-judged relevance.

        Args:
            query: The user's search query
            results: Search results from hybrid search (typically 10-15)

        Returns:
            Top-N results reordered by relevance score
        """
        if not results:
            return []

        if len(results) <= self.top_n:
            return results

        # Build passage list for the prompt
        passages_text = ""
        for i, r in enumerate(results):
            text = r.chunk_text[:500]
            source = r.metadata.get("source_file", "Unknown")
            passages_text += f"[{i}] (Source: {source}) {text}\n\n"

        prompt = RERANK_PROMPT.format(query=query, passages=passages_text)

        try:
            response = self.client.models.generate_content(
                model="gemini-3-flash-preview",
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0,
                    max_output_tokens=256,
                ),
            )

            response_text = response.text.strip()

            # Strip markdown code fences (e.g. ```json ... ```)
            if response_text.startswith("```"):
                lines = response_text.split("\n")
                lines = [l for l in lines if not l.strip().startswith("```")]
                response_text = "\n".join(lines).strip()

            start = response_text.find("[")
            end = response_text.rfind("]") + 1
            if start == -1 or end == 0:
                logger.warning("Reranker returned no valid JSON, returning original order")
                return results[:self.top_n]

            scores = json.loads(response_text[start:end])

            if len(scores) != len(results):
                logger.warning(
                    f"Reranker returned {len(scores)} scores for {len(results)} passages, "
                    "returning original order"
                )
                return results[:self.top_n]

            scored = list(zip(results, scores))
            scored.sort(key=lambda x: x[1], reverse=True)

            reranked = []
            for result, score in scored[:self.top_n]:
                reranked.append(SearchResult(
                    chunk_text=result.chunk_text,
                    document_id=result.document_id,
                    page_number=result.page_number,
                    score=float(score) / 10.0,
                    metadata=result.metadata,
                ))

            return reranked

        except Exception as e:
            logger.error(f"Reranker failed: {e}, returning original order")
            return results[:self.top_n]
