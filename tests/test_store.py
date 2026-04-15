from pathlib import Path

import numpy as np
import pytest

from modern_rag.models import Chunk
from modern_rag.store import LocalIndexStore


def test_store_rejects_index_from_different_embedding_model(tmp_path: Path) -> None:
    store = LocalIndexStore(tmp_path / "index")
    chunks = [Chunk("c1", "doc.txt", "hello", 0, 5)]
    embeddings = np.asarray([[1.0, 0.0]], dtype=np.float32)

    store.save(chunks=chunks, embeddings=embeddings, embedding_model="model-a")

    with pytest.raises(ValueError, match="different embedding model"):
        store.load(expected_embedding_model="model-b")
