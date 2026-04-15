from functools import cached_property

import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    @cached_property
    def model(self) -> SentenceTransformer:
        return SentenceTransformer(self.model_name, device="cpu")

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        payload = [self._format_passage(text) for text in texts]
        vectors = self.model.encode(
            payload,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        return np.asarray(vectors, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        vector = self.model.encode(
            self._format_query(query),
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(vector, dtype=np.float32)

    def _format_passage(self, text: str) -> str:
        if "e5" in self.model_name.lower():
            return f"passage: {text}"
        return text

    def _format_query(self, text: str) -> str:
        if "e5" in self.model_name.lower():
            return f"query: {text}"
        return text

