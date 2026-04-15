import json
from pathlib import Path

import numpy as np

from modern_rag.models import Chunk


class LocalIndexStore:
    def __init__(self, index_path: Path) -> None:
        self.index_path = index_path
        self.index_path.mkdir(parents=True, exist_ok=True)

    @property
    def manifest_file(self) -> Path:
        return self.index_path / "manifest.json"

    @property
    def chunks_file(self) -> Path:
        return self.index_path / "chunks.json"

    @property
    def embeddings_file(self) -> Path:
        return self.index_path / "embeddings.npy"

    def save(self, *, chunks: list[Chunk], embeddings: np.ndarray, embedding_model: str) -> None:
        self.index_path.mkdir(parents=True, exist_ok=True)

        payload = [
            {
                "chunk_id": chunk.chunk_id,
                "source": chunk.source,
                "text": chunk.text,
                "char_start": chunk.char_start,
                "char_end": chunk.char_end,
            }
            for chunk in chunks
        ]
        self.chunks_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        np.save(self.embeddings_file, embeddings)
        self.manifest_file.write_text(
            json.dumps(
                {
                    "embedding_model": embedding_model,
                    "chunks": len(chunks),
                    "embedding_dimension": int(embeddings.shape[1]) if embeddings.ndim == 2 else None,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def load(self, *, expected_embedding_model: str | None = None) -> tuple[list[Chunk], np.ndarray]:
        if not self.exists():
            raise FileNotFoundError("Index files were not found. Run ingestion first.")

        manifest = self.load_manifest()
        if expected_embedding_model and manifest.get("embedding_model") != expected_embedding_model:
            raise ValueError(
                "The stored index was built with a different embedding model. "
                "Run ingestion again to rebuild the index."
            )

        chunks = self.load_chunks()
        embeddings = np.load(self.embeddings_file)
        if embeddings.ndim != 2:
            raise ValueError("Stored embeddings have an invalid shape.")
        if embeddings.shape[0] != len(chunks):
            raise ValueError("Stored embeddings do not match the number of indexed chunks.")
        expected_dimension = manifest.get("embedding_dimension")
        if expected_dimension is not None and embeddings.shape[1] != expected_dimension:
            raise ValueError("Stored embeddings do not match the recorded embedding dimension.")
        return chunks, embeddings

    def load_chunks(self) -> list[Chunk]:
        if not self.chunks_file.exists():
            raise FileNotFoundError("Chunk metadata was not found. Run ingestion first.")

        raw_chunks = json.loads(self.chunks_file.read_text(encoding="utf-8"))
        return [Chunk(**payload) for payload in raw_chunks]

    def load_manifest(self) -> dict:
        if not self.manifest_file.exists():
            raise FileNotFoundError("Index manifest was not found. Run ingestion first.")
        return json.loads(self.manifest_file.read_text(encoding="utf-8"))

    def exists(self) -> bool:
        return self.chunks_file.exists() and self.embeddings_file.exists() and self.manifest_file.exists()
