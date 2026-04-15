import hashlib

from modern_rag.models import Chunk, LoadedDocument


SEPARATORS = ("\n\n", "\n", ". ", " ", "")


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[tuple[int, int, str]]:
    cleaned, start_map, end_map = _normalize_with_span_map(text)
    if not cleaned:
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    chunks: list[tuple[int, int, str]] = []
    cursor = 0
    length = len(cleaned)
    overlap = min(max(chunk_overlap, 0), max(chunk_size - 1, 0))

    while cursor < length:
        end = min(length, cursor + chunk_size)
        if end < length:
            window = cleaned[cursor:end]
            split_at = _find_split_offset(window)
            end = cursor + split_at

        if end <= cursor:
            end = min(length, cursor + chunk_size)

        segment = cleaned[cursor:end]
        left_trim = len(segment) - len(segment.lstrip())
        right_trim = len(segment) - len(segment.rstrip())
        trimmed_start = cursor + left_trim
        trimmed_end = end - right_trim
        snippet = segment.strip()
        if snippet:
            original_start = start_map[trimmed_start]
            original_end = end_map[trimmed_end - 1]
            chunks.append((original_start, original_end, snippet))

        if end == length:
            break

        cursor = max(0, end - overlap)

    return chunks


def chunk_documents(
    documents: list[LoadedDocument],
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    chunks: list[Chunk] = []

    for document in documents:
        for start, end, text in split_text(document.text, chunk_size, chunk_overlap):
            fingerprint = hashlib.sha1(
                f"{document.source}:{start}:{end}:{text}".encode("utf-8")
            ).hexdigest()[:12]
            chunks.append(
                Chunk(
                    chunk_id=fingerprint,
                    source=str(document.source),
                    text=text,
                    char_start=start,
                    char_end=end,
                )
            )

    return chunks


def _find_split_offset(window: str) -> int:
    for separator in SEPARATORS:
        index = window.rfind(separator)
        if index > len(window) * 0.4:
            return index + len(separator)
    return len(window)


def _normalize_with_span_map(text: str) -> tuple[str, list[int], list[int]]:
    normalized_chars: list[str] = []
    start_map: list[int] = []
    end_map: list[int] = []
    pending_space = False
    pending_space_start = 0

    for index, char in enumerate(text):
        if char.isspace():
            if not pending_space:
                pending_space_start = index
            pending_space = bool(normalized_chars)
            continue

        if pending_space:
            normalized_chars.append(" ")
            start_map.append(pending_space_start)
            end_map.append(index)
            pending_space = False

        normalized_chars.append(char)
        start_map.append(index)
        end_map.append(index + 1)

    return "".join(normalized_chars), start_map, end_map
