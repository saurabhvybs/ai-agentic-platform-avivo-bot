import pytest
from storage.db import DBManager


@pytest.fixture
def db(tmp_path):
    """Fresh DBManager for each test — resets singleton so tmp_path DB is used."""
    DBManager.reset()
    db_file = str(tmp_path / "test.db")
    instance = DBManager(db_file)
    yield instance
    DBManager.reset()
