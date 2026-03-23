import hashlib
import json
from datetime import datetime, timezone

from shared.models import RAGResult
from storage.db import DBManager


class QueryCache:
    def __init__(self, db: DBManager, ttl_hours: int) -> None:
        self._db = db
        self._ttl_hours = ttl_hours

    def _hash(self, query_text: str) -> str:
        return hashlib.md5(query_text.lower().strip().encode()).hexdigest()

    def get(self, query_text: str) -> RAGResult | None:
        conn = self._db._get_connection()
        row = conn.execute(
            "SELECT answer, sources, web_refs, created_at FROM query_cache WHERE query_hash = ?",
            (self._hash(query_text),),
        ).fetchone()
        if row is None:
            return None
        created_at = datetime.fromisoformat(str(row["created_at"])).replace(
            tzinfo=timezone.utc
        )
        age_hours = (datetime.now(timezone.utc) - created_at).total_seconds() / 3600
        if age_hours > self._ttl_hours:
            conn.execute("DELETE FROM query_cache WHERE query_hash = ?", (self._hash(query_text),))
            conn.commit()
            return None
        return RAGResult(
            answer=row["answer"],
            sources=json.loads(row["sources"]),
            web_references=json.loads(row["web_refs"]),
            from_cache=True,
        )

    def set(self, query_text: str, result: RAGResult) -> None:
        conn = self._db._get_connection()
        conn.execute(
            """
            INSERT OR REPLACE INTO query_cache
                (query_hash, query_text, answer, sources, web_refs)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                self._hash(query_text),
                query_text,
                result.answer,
                json.dumps(result.sources),
                json.dumps(result.web_references),
            ),
        )
        conn.commit()

    def clear_all(self) -> None:
        conn = self._db._get_connection()
        conn.execute("DELETE FROM query_cache")
        conn.commit()
