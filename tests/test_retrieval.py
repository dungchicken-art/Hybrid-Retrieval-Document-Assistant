import numpy as np

from modern_rag.models import Chunk
from modern_rag.retrieval import HybridRetriever


def test_hybrid_retrieval_surfaces_relevant_chunk() -> None:
    chunks = [
        Chunk("a1", "alpha.txt", "Python powers modern data pipelines.", 0, 36),
        Chunk("b2", "beta.txt", "Oranges and apples are different fruits.", 0, 42),
        Chunk("c3", "gamma.txt", "Hybrid retrieval combines dense and sparse search.", 0, 51),
    ]
    embeddings = np.asarray(
        [
            [0.95, 0.05],
            [0.05, 0.95],
            [0.90, 0.10],
        ],
        dtype=np.float32,
    )

    retriever = HybridRetriever(
        chunks=chunks,
        embeddings=embeddings,
        dense_top_k=2,
        sparse_top_k=2,
        final_top_k=2,
    )
    query_embedding = np.asarray([1.0, 0.0], dtype=np.float32)

    results = retriever.search("How does hybrid retrieval work?", query_embedding)

    assert results
    assert results[0].source == "gamma.txt"
