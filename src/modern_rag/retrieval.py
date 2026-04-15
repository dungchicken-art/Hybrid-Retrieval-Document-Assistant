import re
from dataclasses import dataclass, field

import numpy as np
from rank_bm25 import BM25Okapi

from modern_rag.models import Chunk, SearchResult


TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)


@dataclass(slots=True)
class HybridRetriever:
    chunks: list[Chunk]
    embeddings: np.ndarray
    dense_top_k: int
    sparse_top_k: int
    final_top_k: int
    bm25: BM25Okapi = field(init=False, repr=False)

    def __post_init__(self) -> None:
        corpus = [self._tokenize(chunk.text) for chunk in self.chunks]
        self.bm25 = BM25Okapi(corpus)

    def search(self, query: str, query_embedding: np.ndarray, top_k: int | None = None) -> list[SearchResult]:
        limit = top_k or self.final_top_k

        dense_scores = self.embeddings @ query_embedding
        dense_ranked = np.argsort(-dense_scores)[: self.dense_top_k]

        sparse_scores = self.bm25.get_scores(self._tokenize(query))
        sparse_ranked = np.argsort(-np.asarray(sparse_scores))[: self.sparse_top_k]

        fused = self._rrf(
            dense_ranked.tolist(),
            sparse_ranked.tolist(),
        )

        results: list[SearchResult] = []
        for index, score in fused[:limit]:
            chunk = self.chunks[index]
            results.append(
                SearchResult(
                    chunk_id=chunk.chunk_id,
                    source=chunk.source,
                    score=float(score),
                    text=chunk.text,
                )
            )
        return results

    def _rrf(
        self,
        dense_ranked: list[int],
        sparse_ranked: list[int],
    ) -> list[tuple[int, float]]:
        fused: dict[int, float] = {}
        k = 60

        for rank, index in enumerate(dense_ranked, start=1):
            fused[index] = fused.get(index, 0.0) + (1 / (k + rank))

        for rank, index in enumerate(sparse_ranked, start=1):
            fused[index] = fused.get(index, 0.0) + (1 / (k + rank))

        return sorted(fused.items(), key=lambda item: item[1], reverse=True)

    def _tokenize(self, text: str) -> list[str]:
        return TOKEN_PATTERN.findall(text.lower())
