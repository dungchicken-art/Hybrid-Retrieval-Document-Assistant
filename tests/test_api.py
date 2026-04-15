from pathlib import Path

import pytest

import modern_rag.api as api


def test_resolve_workspace_path_rejects_outside_workspace(tmp_path: Path) -> None:
    original_root = api.workspace_root
    api.workspace_root = tmp_path.resolve()
    try:
        outside = tmp_path.parent / "outside.json"
        with pytest.raises(ValueError, match="project workspace"):
            api._resolve_workspace_path(str(outside))
    finally:
        api.workspace_root = original_root


def test_choose_upload_target_avoids_overwrite(tmp_path: Path) -> None:
    original_docs_path = api.settings.docs_path
    api.settings.docs_path = tmp_path
    try:
        (tmp_path / "notes.md").write_text("first", encoding="utf-8")

        target = api._choose_upload_target("notes.md")

        assert target.name == "notes-1.md"
    finally:
        api.settings.docs_path = original_docs_path
