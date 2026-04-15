import pytest

from modern_rag.evaluation import evaluate_retrieval
from modern_rag.models import EvalCase, SearchResult


class StubPipeline:
    def search(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        mapping = {
            "What is hybrid retrieval?": [
                SearchResult(chunk_id="1", source="demo_data/rag_overview.md", score=1.0, text="hybrid")
            ],
            "Why is a CPU-first design useful for a portfolio project?": [
                SearchResult(chunk_id="9", source="demo_data/rag_overview.md", score=1.0, text="other"),
                SearchResult(chunk_id="2", source="demo_data/cpu_notes.md", score=0.9, text="cpu"),
            ],
            "Why does RAG reduce hallucination?": [
                SearchResult(chunk_id="3", source="demo_data/rag_overview.md", score=1.0, text="grounded")
            ],
        }
        return mapping[query]


def test_evaluate_retrieval_reports_hit_rate() -> None:
    cases = [
        EvalCase(question="What is hybrid retrieval?", expected_sources=["demo_data/rag_overview.md"]),
        EvalCase(
            question="Why is a CPU-first design useful for a portfolio project?",
            expected_sources=["demo_data/cpu_notes.md"],
        ),
        EvalCase(question="Why does RAG reduce hallucination?", expected_sources=["demo_data/rag_overview.md"]),
    ]

    result = evaluate_retrieval(StubPipeline(), cases, top_k=3)

    assert result.total_cases == 3
    assert result.hits_at_k == 3
    assert result.hit_rate == 1.0
    assert result.mrr_at_k == pytest.approx(5 / 6)
    assert result.misses == []
