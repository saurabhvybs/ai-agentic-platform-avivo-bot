from pathlib import Path


class DocumentLoader:
    def load(self, knowledge_base_dir: Path) -> list[tuple[str, str]]:
        """Glob all *.md files, return sorted list of (filename_stem, content)."""
        docs = []
        for path in sorted(knowledge_base_dir.glob("*.md")):
            docs.append((path.stem, path.read_text(encoding="utf-8")))
        return docs
