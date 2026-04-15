from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field


@dataclass(slots=True)
class LoadedDocument:
    source: Path
    text: str


@dataclass(slots=True)
class LoadResult:
    documents: list[LoadedDocument]
    skipped_files: list[str]
    failed_files: list[str]


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    source: str
    text: str
    char_start: int
    char_end: int


class Citation(BaseModel):
    source: str
    chunk_id: str
    score: float
    excerpt: str


class SearchResult(BaseModel):
    chunk_id: str
    source: str
    score: float
    text: str


class SearchRequest(BaseModel):
    query: str = Field(min_length=3)
    top_k: int | None = Field(default=None, ge=1, le=20)


class AskRequest(BaseModel):
    query: str = Field(min_length=3)
    top_k: int | None = Field(default=None, ge=1, le=20)


class IngestRequest(BaseModel):
    source: str | None = None


class IngestResponse(BaseModel):
    documents: int
    chunks: int
    index_path: str
    skipped_files: list[str] = Field(default_factory=list)
    failed_files: list[str] = Field(default_factory=list)


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    mode: str = "retrieval_only"
    warning: str | None = None


class SourceSummary(BaseModel):
    source: str
    chunk_count: int


class EvalCase(BaseModel):
    question: str = Field(min_length=3)
    expected_sources: list[str] = Field(min_length=1)


class EvalRequest(BaseModel):
    dataset_path: str
    top_k: int | None = Field(default=None, ge=1, le=20)


class EvalResponse(BaseModel):
    total_cases: int
    hit_rate: float
    hits_at_k: int
    mrr_at_k: float
    misses: list[str]
