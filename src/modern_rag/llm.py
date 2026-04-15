import asyncio

import httpx

from modern_rag.models import AskResponse, Citation, SearchResult


class LLMClient:
    def __init__(
        self,
        *,
        base_url: str | None,
        api_key: str | None,
        model: str | None,
        max_context_chunks: int,
    ) -> None:
        self.base_url = base_url.rstrip("/") if base_url else None
        self.api_key = api_key
        self.model = model
        self.max_context_chunks = max_context_chunks

    @property
    def enabled(self) -> bool:
        return bool(self.base_url and self.api_key and self.model)

    async def answer(self, query: str, results: list[SearchResult]) -> AskResponse:
        citations = [
            Citation(
                source=result.source,
                chunk_id=result.chunk_id,
                score=result.score,
                excerpt=result.text[:280],
            )
            for result in results[: self.max_context_chunks]
        ]

        if not self.enabled:
            answer = self._fallback_answer(query, citations)
            return AskResponse(
                answer=answer,
                citations=citations,
                mode="retrieval_only",
                warning="No LLM endpoint is configured.",
            )

        prompt = self._build_prompt(query, results[: self.max_context_chunks])
        payload = {
            "model": self.model,
            "temperature": 0.1,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You answer using only the supplied context. "
                        "If the answer is not supported by the context, say that clearly. "
                        "Always cite source file names inline."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            content = await self._request_completion(headers=headers, payload=payload)
            return AskResponse(answer=content.strip(), citations=citations, mode="llm")
        except Exception as exc:
            warning = self._friendly_error_message(exc)
            answer = self._fallback_answer(query, citations)
            return AskResponse(
                answer=answer,
                citations=citations,
                mode="retrieval_only",
                warning=warning,
            )

    def _build_prompt(self, query: str, results: list[SearchResult]) -> str:
        context = []
        for index, result in enumerate(results, start=1):
            context.append(
                f"[{index}] Source: {result.source}\n"
                f"Chunk ID: {result.chunk_id}\n"
                f"Content: {result.text}"
            )
        return f"Question: {query}\n\nContext:\n\n" + "\n\n".join(context)

    def _fallback_answer(self, query: str, citations: list[Citation]) -> str:
        if not citations:
            return "No relevant context was found."

        lines = [
            "No LLM endpoint is configured, so this is a retrieval-only response.",
            f"Question: {query}",
            "Most relevant sources:",
        ]
        for citation in citations:
            lines.append(f"- {citation.source}: {citation.excerpt}")
        return "\n".join(lines)

    async def _request_completion(self, *, headers: dict[str, str], payload: dict) -> str:
        last_error: Exception | None = None

        for attempt in range(3):
            try:
                async with httpx.AsyncClient(timeout=180) as client:
                    response = await client.post(
                        f"{self.base_url}/chat/completions",
                        headers=headers,
                        json=payload,
                    )
                    response.raise_for_status()
                    return response.json()["choices"][0]["message"]["content"]
            except httpx.HTTPStatusError as exc:
                last_error = exc
                status_code = exc.response.status_code
                if status_code == 429 and attempt < 2:
                    await asyncio.sleep(1.5 * (attempt + 1))
                    continue
                raise
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                last_error = exc
                if attempt < 2:
                    await asyncio.sleep(1.5 * (attempt + 1))
                    continue
                raise

        if last_error is not None:
            raise last_error
        raise RuntimeError("LLM request failed without an error.")

    def _friendly_error_message(self, exc: Exception) -> str:
        if isinstance(exc, httpx.HTTPStatusError):
            status_code = exc.response.status_code
            if status_code == 401:
                return "LLM request failed: invalid API key or unauthorized project."
            if status_code == 404:
                return "LLM request failed: endpoint or model was not found."
            if status_code == 429:
                return "LLM request failed: rate limit or quota exceeded. Falling back to retrieval-only mode."
            if 500 <= status_code < 600:
                return "LLM request failed: upstream model server is unavailable."
            return f"LLM request failed with HTTP {status_code}."

        if isinstance(exc, httpx.TimeoutException):
            return "LLM request timed out. Falling back to retrieval-only mode."

        if isinstance(exc, httpx.NetworkError):
            return "LLM request failed due to a network error. Falling back to retrieval-only mode."

        return f"LLM request failed: {exc}"
