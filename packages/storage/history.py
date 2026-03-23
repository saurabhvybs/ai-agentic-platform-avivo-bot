from datetime import datetime, timezone

from shared.models import HistoryEntry
from storage.db import DBManager


class ConversationHistory:
    def __init__(self, db: DBManager, max_history: int) -> None:
        self._db = db
        self._max_history = max_history

    def _row_to_entry(self, row) -> HistoryEntry:
        return HistoryEntry(
            user_id=row["user_id"],
            role=row["role"],
            content=row["content"],
            created_at=datetime.fromisoformat(str(row["created_at"])).replace(
                tzinfo=timezone.utc
            ),
        )

    def get(self, user_id: int) -> list[HistoryEntry]:
        """Return last MAX_HISTORY entries, oldest-first."""
        conn = self._db._get_connection()
        rows = conn.execute(
            """
            SELECT user_id, role, content, created_at
            FROM conversation_history
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (user_id, self._max_history),
        ).fetchall()
        return [self._row_to_entry(r) for r in reversed(rows)]

    def add(self, user_id: int, role: str, content: str) -> None:
        """Insert message and trim to last MAX_HISTORY per user.

        Safe from race conditions: single-connection singleton serializes all writes.
        """
        conn = self._db._get_connection()
        conn.execute(
            "INSERT INTO conversation_history (user_id, role, content) VALUES (?, ?, ?)",
            (user_id, role, content),
        )
        conn.execute(
            """
            DELETE FROM conversation_history
            WHERE user_id = ? AND id NOT IN (
                SELECT id FROM conversation_history
                WHERE user_id = ?
                ORDER BY id DESC
                LIMIT ?
            )
            """,
            (user_id, user_id, self._max_history),
        )
        conn.commit()

    def get_all_for_summary(self, user_id: int) -> list[HistoryEntry]:
        """Return up to 50 most-recent entries for user, oldest-first (used by /summarize).

        Capped to prevent unbounded LLM context and runaway API costs.
        """
        conn = self._db._get_connection()
        rows = conn.execute(
            """
            SELECT user_id, role, content, created_at
            FROM conversation_history
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT 50
            """,
            (user_id,),
        ).fetchall()
        return [self._row_to_entry(r) for r in reversed(rows)]
