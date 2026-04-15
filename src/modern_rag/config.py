from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = Field(default="Modern RAG CPU", alias="RAG_APP_NAME")
    docs_path: Path = Field(default=Path("knowledge"), alias="RAG_DOCS_PATH")
    index_path: Path = Field(default=Path("data/index"), alias="RAG_INDEX_PATH")
    embedding_model: str = Field(
        default="intfloat/multilingual-e5-small",
        alias="RAG_EMBEDDING_MODEL",
    )
    chunk_size: int = Field(default=900, alias="RAG_CHUNK_SIZE")
    chunk_overlap: int = Field(default=150, alias="RAG_CHUNK_OVERLAP")
    dense_top_k: int = Field(default=6, alias="RAG_DENSE_TOP_K")
    sparse_top_k: int = Field(default=6, alias="RAG_SPARSE_TOP_K")
    final_top_k: int = Field(default=5, alias="RAG_FINAL_TOP_K")
    max_context_chunks: int = Field(default=4, alias="RAG_MAX_CONTEXT_CHUNKS")
    openai_base_url: str | None = Field(default=None, alias="OPENAI_BASE_URL")
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_model: str | None = Field(default=None, alias="OPENAI_MODEL")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

