import asyncio

import httpx

from modern_rag.llm import LLMClient
from modern_rag.models import SearchResult


class RateLimitedLLMClient(LLMClient):
    async def _request_completion(self, *, headers: dict[str, str], payload: dict) -> str:
        request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        response = httpx.Response(429, request=request)
        raise httpx.HTTPStatusError("rate limited", request=request, response=response)


def test_llm_falls_back_to_retrieval_on_rate_limit() -> None:
    client = RateLimitedLLMClient(
        base_url="https://api.openai.com/v1",
        api_key="test-key",
        model="gpt-4.1-mini",
        max_context_chunks=2,
    )
    results = [
        SearchResult(
            chunk_id="abc123",
            source="demo_data/rag_overview.md",
            score=0.9,
            text="Hybrid retrieval combines dense and sparse ranking.",
        )
    ]

    response = asyncio.run(client.answer("What is hybrid retrieval?", results))

    assert response.mode == "retrieval_only"
    assert response.warning is not None
    assert "rate limit or quota exceeded" in response.warning
    assert response.citations
