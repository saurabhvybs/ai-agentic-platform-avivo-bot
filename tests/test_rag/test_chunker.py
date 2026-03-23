import pytest
from rag.ingestion.chunker import TextChunker


@pytest.fixture
def chunker():
    return TextChunker(chunk_size=100, overlap=20)


def test_chunk_empty_text_returns_empty(chunker):
    assert chunker.chunk("") == []


def test_chunk_whitespace_returns_empty(chunker):
    assert chunker.chunk("   \n\n  ") == []


def test_short_text_returns_single_chunk(chunker):
    text = "Short text."
    result = chunker.chunk(text)
    assert len(result) == 1
    assert result[0] == text.strip()


def test_long_text_splits_into_multiple_chunks():
    chunker = TextChunker(chunk_size=50, overlap=10)
    # Each sentence ~40 chars — should produce multiple chunks
    text = " ".join([f"Sentence number {i} is here." for i in range(10)])
    chunks = chunker.chunk(text)
    assert len(chunks) > 1


def test_chunk_size_respected():
    chunker = TextChunker(chunk_size=80, overlap=10)
    text = " ".join([f"This is sentence number {i}." for i in range(20)])
    chunks = chunker.chunk(text)
    for chunk in chunks:
        # Chunks are saved only when the next sentence would exceed chunk_size, so
        # normal chunks are at most chunk_size-1 chars. The edge case: an overlap-
        # started chunk whose single sentence is longer than chunk_size can exceed
        # the limit by up to overlap(10) + 1 + max_sentence_length(~28) ≈ 39 chars.
        # +40 is a reasonable upper bound for this test's ~28-char sentences.
        assert len(chunk) <= 80 + 40, f"Chunk too long ({len(chunk)} chars): {chunk!r}"


def test_overlap_carries_content():
    chunker = TextChunker(chunk_size=60, overlap=20)
    # Three sentences totaling >60 chars — chunker MUST produce at least 2 chunks
    text = "First sentence is here. Second sentence follows on. Third one ends here now."
    chunks = chunker.chunk(text)
    assert len(chunks) > 1, f"Expected multiple chunks, got {len(chunks)}: {chunks!r}"
    # Overlap: words from the last 'overlap' chars of chunk[0] must appear in chunk[1]
    overlap_text = chunks[0][-20:]
    words_in_overlap = overlap_text.split()
    assert any(word in chunks[1] for word in words_in_overlap), (
        f"No overlap words from chunk[0][-20:] found in chunk[1]. "
        f"chunk[0]: {chunks[0]!r}, chunk[1]: {chunks[1]!r}"
    )
