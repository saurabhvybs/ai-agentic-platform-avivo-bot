import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from telegram import Update, User, Message
from bot.middleware.allowlist import allowed


def _make_update(user_id: int):
    user = MagicMock(spec=User)
    user.id = user_id
    message = MagicMock(spec=Message)
    message.reply_text = AsyncMock()
    update = MagicMock(spec=Update)
    update.effective_user = user
    update.message = message
    return update


@pytest.fixture
def context():
    return MagicMock()


async def test_allowed_user_passes_through():
    update = _make_update(123)
    handler_called = False

    with patch("bot.middleware.allowlist.settings") as mock_settings:
        mock_settings.ALLOWED_USER_IDS = [123, 456]

        @allowed
        async def handler(u, c):
            nonlocal handler_called
            handler_called = True

        await handler(update, MagicMock())

    assert handler_called


async def test_blocked_user_gets_denied_reply():
    update = _make_update(999)

    with patch("bot.middleware.allowlist.settings") as mock_settings:
        mock_settings.ALLOWED_USER_IDS = [123]

        @allowed
        async def handler(u, c):
            pass

        await handler(update, MagicMock())

    update.message.reply_text.assert_called_once()
    args = update.message.reply_text.call_args[0]
    assert "not authorized" in args[0].lower()


async def test_empty_allowlist_allows_all():
    """Empty ALLOWED_USER_IDS means open bot — any user passes through."""
    update = _make_update(123)
    handler_called = False

    with patch("bot.middleware.allowlist.settings") as mock_settings:
        mock_settings.ALLOWED_USER_IDS = []

        @allowed
        async def handler(u, c):
            nonlocal handler_called
            handler_called = True

        await handler(update, MagicMock())

    assert handler_called
    update.message.reply_text.assert_not_called()


async def test_blocked_user_does_not_call_handler():
    update = _make_update(999)
    handler_called = False

    with patch("bot.middleware.allowlist.settings") as mock_settings:
        mock_settings.ALLOWED_USER_IDS = [123]

        @allowed
        async def handler(u, c):
            nonlocal handler_called
            handler_called = True

        await handler(update, MagicMock())

    assert not handler_called
