from modern_rag.chunking import split_text


def test_split_text_creates_multiple_chunks_with_overlap() -> None:
    text = " ".join(f"token{i}" for i in range(100))

    chunks = split_text(text, chunk_size=80, chunk_overlap=20)

    assert len(chunks) > 1
    assert chunks[0][0] == 0
    assert chunks[1][0] < chunks[0][1]


def test_split_text_handles_overlap_larger_than_chunk_size() -> None:
    text = "A " * 200

    chunks = split_text(text, chunk_size=40, chunk_overlap=100)

    assert len(chunks) > 1
    assert all(start < end for start, end, _ in chunks)


def test_split_text_preserves_offsets_in_original_text() -> None:
    text = "Alpha\n\nBeta   Gamma"

    chunks = split_text(text, chunk_size=64, chunk_overlap=0)

    assert chunks == [(0, len(text), "Alpha Beta Gamma")]


def test_split_text_preserves_offsets_after_internal_whitespace_collapse() -> None:
    text = "Alpha   Beta"

    chunks = split_text(text, chunk_size=6, chunk_overlap=0)

    assert chunks[0] == (0, 5, "Alpha")
    assert chunks[1] == (8, 12, "Beta")
