from pathlib import Path
from rag.ingestion.loader import DocumentLoader


def test_load_returns_stem_and_content(tmp_path):
    (tmp_path / "policy.md").write_text("# Policy\nSome content.")
    loader = DocumentLoader()
    docs = loader.load(tmp_path)
    assert len(docs) == 1
    stem, content = docs[0]
    assert stem == "policy"
    assert "Some content" in content


def test_load_multiple_files_sorted(tmp_path):
    (tmp_path / "b_doc.md").write_text("B content")
    (tmp_path / "a_doc.md").write_text("A content")
    loader = DocumentLoader()
    docs = loader.load(tmp_path)
    stems = [d[0] for d in docs]
    assert stems == sorted(stems)


def test_load_ignores_non_md_files(tmp_path):
    (tmp_path / "policy.md").write_text("md content")
    (tmp_path / "notes.txt").write_text("txt content")
    loader = DocumentLoader()
    docs = loader.load(tmp_path)
    assert len(docs) == 1


def test_load_empty_directory(tmp_path):
    loader = DocumentLoader()
    assert loader.load(tmp_path) == []
