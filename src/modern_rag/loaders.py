import json
from pathlib import Path

from pypdf import PdfReader

from modern_rag.models import LoadResult, LoadedDocument


SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".json"}


def load_documents(source_dir: Path) -> LoadResult:
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")

    documents: list[LoadedDocument] = []
    skipped_files: list[str] = []
    failed_files: list[str] = []
    for path in sorted(source_dir.rglob("*")):
        if path.is_dir() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        try:
            text = _read_file(path)
        except Exception as exc:
            failed_files.append(f"{path}: {exc}")
            continue

        if text.strip():
            documents.append(LoadedDocument(source=path, text=text))
        else:
            skipped_files.append(f"{path}: no extractable text")

    return LoadResult(
        documents=documents,
        skipped_files=skipped_files,
        failed_files=failed_files,
    )


def _read_file(path: Path) -> str:
    suffix = path.suffix.lower()

    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="ignore")

    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
        return json.dumps(payload, ensure_ascii=False, indent=2)

    if suffix == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    raise ValueError(f"Unsupported file type: {path}")
