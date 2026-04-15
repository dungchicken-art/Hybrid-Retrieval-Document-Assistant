from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from modern_rag.config import get_settings
from modern_rag.evaluation import evaluate_retrieval, load_eval_cases
from modern_rag.models import (
    AskRequest,
    AskResponse,
    EvalRequest,
    EvalResponse,
    IngestRequest,
    IngestResponse,
    SearchRequest,
    SearchResult,
    SourceSummary,
)
from modern_rag.pipeline import RagPipeline


settings = get_settings()
pipeline = RagPipeline(settings)
workspace_root = Path.cwd().resolve()
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
UPLOAD_CHUNK_BYTES = 1024 * 1024


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield


app = FastAPI(title=settings.app_name, lifespan=lifespan)
web_dir = Path(__file__).parent / "web"
app.mount("/assets", StaticFiles(directory=web_dir), name="assets")


@app.get("/", include_in_schema=False)
async def home() -> FileResponse:
    return FileResponse(web_dir / "index.html")


@app.get("/health")
async def health() -> dict[str, bool | str]:
    return {
        "app": settings.app_name,
        "index_ready": pipeline.has_index(),
        "llm_configured": pipeline.llm.enabled,
    }


@app.post("/ingest", response_model=IngestResponse)
async def ingest(request: IngestRequest) -> IngestResponse:
    try:
        source = _resolve_workspace_path(request.source) if request.source else None
        return pipeline.ingest(source=source)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/search", response_model=list[SearchResult])
async def search(request: SearchRequest) -> list[SearchResult]:
    try:
        return pipeline.search(query=request.query, top_k=request.top_k)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest) -> AskResponse:
    try:
        return await pipeline.ask(query=request.query, top_k=request.top_k)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/sources", response_model=list[SourceSummary])
async def sources() -> list[SourceSummary]:
    try:
        return pipeline.list_sources()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/upload")
async def upload(files: list[UploadFile] = File(...)) -> dict[str, list[str]]:
    try:
        saved: list[str] = []
        settings.docs_path.mkdir(parents=True, exist_ok=True)
        allowed = {".txt", ".md", ".pdf", ".json"}

        for file in files:
            if not file.filename:
                continue

            suffix = Path(file.filename).suffix.lower()
            if suffix not in allowed:
                raise ValueError(f"Unsupported file type: {file.filename}")

            target = _choose_upload_target(Path(file.filename).name)
            bytes_written = 0
            try:
                with target.open("wb") as handle:
                    while True:
                        chunk = await file.read(UPLOAD_CHUNK_BYTES)
                        if not chunk:
                            break
                        bytes_written += len(chunk)
                        if bytes_written > MAX_UPLOAD_BYTES:
                            raise ValueError(
                                f"Upload exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit: {file.filename}"
                            )
                        handle.write(chunk)
            except Exception:
                target.unlink(missing_ok=True)
                raise
            saved.append(str(target))

        return {"saved_files": saved}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/eval", response_model=EvalResponse)
async def evaluate(request: EvalRequest) -> EvalResponse:
    try:
        cases = load_eval_cases(_resolve_workspace_path(request.dataset_path))
        return evaluate_retrieval(pipeline, cases, top_k=request.top_k)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _resolve_workspace_path(raw_path: str) -> Path:
    candidate = Path(raw_path)
    resolved = candidate.resolve() if candidate.is_absolute() else (workspace_root / candidate).resolve()
    if workspace_root not in resolved.parents and resolved != workspace_root:
        raise ValueError("Path must stay inside the project workspace.")
    return resolved


def _choose_upload_target(filename: str) -> Path:
    safe_name = Path(filename).name
    target = settings.docs_path / safe_name
    if not target.exists():
        return target

    stem = target.stem
    suffix = target.suffix
    counter = 1
    while True:
        candidate = settings.docs_path / f"{stem}-{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1
