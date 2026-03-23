# tests/test_storage/test_history.py
import pytest
from storage.history import ConversationHistory


@pytest.fixture
def history(db):
    return ConversationHistory(db, max_history=3)


def test_get_empty_returns_empty_list(history):
    assert history.get(999) == []


def test_add_and_get_returns_messages(history):
    history.add(1, "user", "Hello")
    history.add(1, "assistant", "Hi there")
    entries = history.get(1)
    assert len(entries) == 2
    assert entries[0].role == "user"
    assert entries[1].role == "assistant"


def test_get_returns_chronological_order(history):
    for i in range(3):
        history.add(1, "user", f"msg {i}")
    entries = history.get(1)
    contents = [e.content for e in entries]
    assert contents == ["msg 0", "msg 1", "msg 2"]


def test_add_trims_to_max_history(history):
    for i in range(5):
        history.add(1, "user", f"msg {i}")
    entries = history.get(1)
    assert len(entries) == 3
    # Most recent 3
    contents = [e.content for e in entries]
    assert contents == ["msg 2", "msg 3", "msg 4"]


def test_get_isolates_by_user_id(history):
    history.add(1, "user", "user 1 msg")
    history.add(2, "user", "user 2 msg")
    assert len(history.get(1)) == 1
    assert len(history.get(2)) == 1


def test_get_all_for_summary_returns_all(db):
    # Use max_history=10 so add() does NOT trim — tests that get_all_for_summary
    # returns every row, not just the max_history window.
    # (With max_history=3, add() trims to 3 rows after each insert, so only 3 survive.)
    history = ConversationHistory(db, max_history=10)
    for i in range(5):
        history.add(1, "user", f"msg {i}")
    all_entries = history.get_all_for_summary(1)
    assert len(all_entries) == 5
