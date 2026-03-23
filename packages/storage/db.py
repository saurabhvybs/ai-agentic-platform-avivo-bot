import sqlite3
from pathlib import Path

import sqlite_vec


class DBManager:
    _instance: "DBManager | None" = None
    _connection: sqlite3.Connection | None = None

    @classmethod
    def get_instance(cls, db_path: str = ".db/genai_bot.db") -> "DBManager":
        if cls._instance is None:
            cls._instance = cls(db_path)
        return cls._instance

    def __init__(self, db_path: str = ".db/genai_bot.db") -> None:
        # Guard: prevent double-init if constructor called directly after get_instance
        if self.__class__._connection is not None:
            return
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        conn.execute("PRAGMA journal_mode=WAL")
        self.__class__._connection = conn
        self._run_schema()

    def _run_schema(self) -> None:
        schema_path = Path(__file__).parent / "schema.sql"
        self.__class__._connection.executescript(schema_path.read_text())
        self.__class__._connection.commit()

    def _get_connection(self) -> sqlite3.Connection:
        """Private — only call from within storage/ module methods."""
        return self.__class__._connection

    def close(self) -> None:
        """Close the connection and clear singleton state."""
        if self.__class__._connection:
            self.__class__._connection.close()
        self.__class__._connection = None
        self.__class__._instance = None

    @classmethod
    def reset(cls) -> None:
        """For tests only — reset singleton so a fresh DB path can be used."""
        if cls._connection:
            try:
                cls._connection.close()
            except Exception:
                pass
        cls._connection = None
        cls._instance = None
