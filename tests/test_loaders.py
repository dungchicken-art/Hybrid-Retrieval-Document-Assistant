from pathlib import Path

from modern_rag.loaders import load_documents


def test_load_documents_continues_when_one_file_fails(tmp_path: Path) -> None:
    (tmp_path / "good.md").write_text("# Hello", encoding="utf-8")
    (tmp_path / "bad.json").write_text("{not-json}", encoding="utf-8")

    result = load_documents(tmp_path)

    assert len(result.documents) == 1
    assert result.documents[0].source.name == "good.md"
    assert result.failed_files
    assert "bad.json" in result.failed_files[0]
