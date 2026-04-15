import json
from pathlib import Path, PurePath

from modern_rag.models import EvalCase, EvalResponse
from modern_rag.pipeline import RagPipeline


def load_eval_cases(dataset_path: Path) -> list[EvalCase]:
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    return [EvalCase(**row) for row in payload]


def evaluate_retrieval(
    pipeline: RagPipeline,
    cases: list[EvalCase],
    *,
    top_k: int | None = None,
) -> EvalResponse:
    misses: list[str] = []
    hits = 0
    reciprocal_rank_sum = 0.0

    for case in cases:
        results = pipeline.search(case.question, top_k=top_k)
        expected_sources = {_normalize_source(source) for source in case.expected_sources}
        rank = None

        for index, result in enumerate(results, start=1):
            if _normalize_source(result.source) in expected_sources:
                rank = index
                break

        if rank is not None:
            hits += 1
            reciprocal_rank_sum += 1 / rank
        else:
            misses.append(case.question)

    total = len(cases)
    hit_rate = hits / total if total else 0.0
    mrr_at_k = reciprocal_rank_sum / total if total else 0.0
    return EvalResponse(
        total_cases=total,
        hit_rate=hit_rate,
        hits_at_k=hits,
        mrr_at_k=mrr_at_k,
        misses=misses,
    )


def _normalize_source(source: str) -> str:
    return PurePath(source).as_posix().lower()
