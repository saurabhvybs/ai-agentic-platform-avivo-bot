from storage.db import DBManager


def test_schema_creates_all_tables(db):
    conn = db._get_connection()
    tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    assert "document_chunks" in tables
    assert "query_cache" in tables
    assert "conversation_history" in tables


def test_wal_mode_enabled(db):
    conn = db._get_connection()
    mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode == "wal"


def test_sqlite_vec_extension_loaded(db):
    conn = db._get_connection()
    # sqlite-vec adds vec_version() function
    row = conn.execute("SELECT vec_version()").fetchone()
    assert row is not None


def test_index_on_history_user_id(db):
    conn = db._get_connection()
    indexes = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index'"
        ).fetchall()
    }
    assert "idx_history_user" in indexes


def test_singleton_returns_same_instance(tmp_path):
    DBManager.reset()
    db_file = str(tmp_path / "singleton.db")
    a = DBManager.get_instance(db_file)
    b = DBManager.get_instance(db_file)
    assert a is b
    DBManager.reset()


def test_reset_allows_fresh_instance(tmp_path):
    DBManager.reset()
    db1 = DBManager(str(tmp_path / "db1.db"))
    DBManager.reset()
    db2 = DBManager(str(tmp_path / "db2.db"))
    assert db1 is not db2
    DBManager.reset()
