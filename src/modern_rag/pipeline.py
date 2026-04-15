from pathlib import Path

from modern_rag.chunking import chunk_documents
from modern_rag.config import Settings
from modern_rag.embeddings import Embedder
from modern_rag.llm import LLMClient
from modern_rag.loaders import load_documents
from modern_rag.models import AskResponse, IngestResponse, SearchResult, SourceSummary
from modern_rag.retrieval import HybridRetriever
from modern_rag.store import LocalIndexStore


class RagPipeline:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.embedder = Embedder(settings.embedding_model)
        self.store = LocalIndexStore(settings.index_path)
        self.llm = LLMClient(
            base_url=settings.openai_base_url,
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            max_context_chunks=settings.max_context_chunks,
        )
        self._retriever: HybridRetriever | None = None

    def ingest(self, source: Path | None = None) -> IngestResponse:
        source_dir = source or self.settings.docs_path
        load_result = load_documents(source_dir)
        documents = load_result.documents
        chunks = chunk_documents(
            documents,
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )
        if not chunks:
            summary = []
            if load_result.failed_files:
                summary.append(f"{len(load_result.failed_files)} files failed to load")
            if load_result.skipped_files:
                summary.append(f"{len(load_result.skipped_files)} files had no extractable text")
            detail = f" ({', '.join(summary)})" if summary else ""
            raise ValueError(f"No chunks were created from {source_dir}{detail}")

        embeddings = self.embedder.embed_documents([chunk.text for chunk in chunks])
        self.store.save(
            chunks=chunks,
            embeddings=embeddings,
            embedding_model=self.settings.embedding_model,
        )
        self._retriever = HybridRetriever(
            chunks=chunks,
            embeddings=embeddings,
            dense_top_k=self.settings.dense_top_k,
            sparse_top_k=self.settings.sparse_top_k,
            final_top_k=self.settings.final_top_k,
        )
        return IngestResponse(
            documents=len(documents),
            chunks=len(chunks),
            index_path=str(self.settings.index_path),
            skipped_files=load_result.skipped_files,
            failed_files=load_result.failed_files,
        )

    def search(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        retriever = self._get_retriever()
        query_embedding = self.embedder.embed_query(query)
        return retriever.search(query=query, query_embedding=query_embedding, top_k=top_k)

    async def ask(self, query: str, top_k: int | None = None) -> AskResponse:
        results = self.search(query=query, top_k=top_k)
        return await self.llm.answer(query=query, results=results)

    def has_index(self) -> bool:
        return self.store.exists()

    def list_sources(self) -> list[SourceSummary]:
        chunks = self.store.load_chunks()
        counts: dict[str, int] = {}
        for chunk in chunks:
            counts[chunk.source] = counts.get(chunk.source, 0) + 1

        return [
            SourceSummary(source=source, chunk_count=count)
            for source, count in sorted(counts.items())
        ]

    def _get_retriever(self) -> HybridRetriever:
        if self._retriever is not None:
            return self._retriever

        chunks, embeddings = self.store.load(expected_embedding_model=self.settings.embedding_model)
        self._retriever = HybridRetriever(
            chunks=chunks,
            embeddings=embeddings,
            dense_top_k=self.settings.dense_top_k,
            sparse_top_k=self.settings.sparse_top_k,
            final_top_k=self.settings.final_top_k,
        )
        return self._retriever
