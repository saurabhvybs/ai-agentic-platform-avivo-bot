# tests/test_shared_config.py
import logging
from shared.config import Settings
from shared.logger import UserIdFilter, configure_logging, user_id_context


def test_allowed_user_ids_parsed_from_comma_string():
    result = Settings.parse_allowed_user_ids("123,456,789")
    assert result == [123, 456, 789]


def test_allowed_user_ids_handles_empty_string():
    result = Settings.parse_allowed_user_ids("")
    assert result == []


def test_allowed_user_ids_handles_spaces():
    result = Settings.parse_allowed_user_ids(" 123 , 456 ")
    assert result == [123, 456]


def test_user_id_filter_adds_user_id_to_log_record():
    user_id_context.set("42")
    f = UserIdFilter()
    record = logging.LogRecord("test", logging.INFO, "", 0, "msg", (), None)
    f.filter(record)
    assert record.user_id == "42"


def test_configure_logging_does_not_raise():
    configure_logging("DEBUG")  # Should not raise
