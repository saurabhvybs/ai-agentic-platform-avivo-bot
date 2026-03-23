import sqlite_vec

from storage.db import DBManager


class VectorStore:
    def __init__(self, db: DBManager) -> None:
        self._db = db

    def insert(
        self, doc_name: str, chunk_index: int, text: str, embedding: list[float]
    ) -> None:
        conn = self._db._get_connection()
        cursor = conn.execute(
            "INSERT INTO document_chunks (doc_name, chunk_index, text) VALUES (?, ?, ?)",
            (doc_name, chunk_index, text),
        )
        rowid = cursor.lastrowid
        conn.execute(
            "INSERT INTO chunk_embeddings (rowid, embedding) VALUES (?, ?)",
            (rowid, sqlite_vec.serialize_float32(embedding)),
        )
        conn.commit()

    def search(self, vector: list[float], top_k: int) -> list[tuple[str, str]]:
        conn = self._db._get_connection()
        rows = conn.execute(
            """
            SELECT dc.text, dc.doc_name
            FROM chunk_embeddings ce
            JOIN document_chunks dc ON dc.id = ce.rowid
            WHERE ce.embedding MATCH ?
              AND k = ?
            ORDER BY distance
            """,
            (sqlite_vec.serialize_float32(vector), top_k),
        ).fetchall()
        return [(row["text"], row["doc_name"]) for row in rows]

    def delete_by_doc(self, doc_name: str) -> None:
        conn = self._db._get_connection()
        with conn:  # BEGIN / COMMIT or ROLLBACK — both deletes are atomic
            rowids = [
                row[0]
                for row in conn.execute(
                    "SELECT id FROM document_chunks WHERE doc_name = ?", (doc_name,)
                ).fetchall()
            ]
            if rowids:
                conn.executemany(
                    "DELETE FROM chunk_embeddings WHERE rowid = ?",
                    [(r,) for r in rowids],
                )
            conn.execute("DELETE FROM document_chunks WHERE doc_name = ?", (doc_name,))

    def has_doc(self, doc_name: str) -> bool:
        conn = self._db._get_connection()
        row = conn.execute(
            "SELECT COUNT(*) FROM document_chunks WHERE doc_name = ?", (doc_name,)
        ).fetchone()
        return row[0] > 0

    def has_doc_any(self) -> bool:
        """Return True if the knowledge base has at least one chunk ingested."""
        conn = self._db._get_connection()
        row = conn.execute("SELECT COUNT(*) FROM document_chunks").fetchone()
        return row[0] > 0
