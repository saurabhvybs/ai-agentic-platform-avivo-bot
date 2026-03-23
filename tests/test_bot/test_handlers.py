# tests/test_bot/test_handlers.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from telegram import Update, User, Message, PhotoSize

from shared.models import RAGResult, VisionResult


@pytest.fixture(autouse=True)
def clear_rate_limit_state():
    """Clear sliding-window rate limiter state between handler tests."""
    from bot.middleware import rate_limit as rl
    rl._user_timestamps.clear()
    yield
    rl._user_timestamps.clear()


def _make_update(user_id: int = 123, text: str = ""):
    user = MagicMock(spec=User)
    user.id = user_id
    message = MagicMock(spec=Message)
    message.reply_text = AsyncMock()
    message.chat.send_action = AsyncMock()
    message.photo = None
    update = MagicMock(spec=Update)
    update.effective_user = user
    update.message = message
    return update


def _make_context(args=None):
    ctx = MagicMock()
    ctx.args = args or []
    return ctx


# --- help handler ---

async def test_help_handler_replies_with_help_text():
    update = _make_update(123)
    # Both patches must be active when help_handler executes
    with (
        patch("bot.middleware.allowlist.settings") as s1,
        patch("bot.middleware.rate_limit.settings") as s2,
    ):
        s1.ALLOWED_USER_IDS = [123]
        s2.RATE_LIMIT_PER_MINUTE = 100
        from bot.handlers.help import help_handler
        await help_handler(update, _make_context())
    update.message.reply_text.assert_called_once()
    assert "/ask" in update.message.reply_text.call_args[0][0]


# --- ask handler ---

async def test_ask_handler_empty_query_replies_with_usage():
    update = _make_update(123)
    from bot.handlers import ask
    with (
        patch("bot.middleware.allowlist.settings") as s1,
        patch("bot.middleware.rate_limit.settings") as s2,
        patch.object(ask, "_query_cache") as mock_cache,
    ):
        s1.ALLOWED_USER_IDS = [123]
        s2.RATE_LIMIT_PER_MINUTE = 100
        mock_cache.get.return_value = None
        from bot.handlers.ask import ask_handler
        await ask_handler(update, _make_context(args=[]))
    update.message.reply_text.assert_called_once()
    assert "Please provide" in update.message.reply_text.call_args[0][0]


async def test_ask_handler_returns_cached_result():
    update = _make_update(123)
    cached = RAGResult(answer="Cached answer", sources=["doc.md"], web_references=[], from_cache=True)
    from bot.handlers import ask
    with (
        patch("bot.middleware.allowlist.settings") as s1,
        patch("bot.middleware.rate_limit.settings") as s2,
        patch.object(ask, "_query_cache") as mock_cache,
    ):
        s1.ALLOWED_USER_IDS = [123]
        s2.RATE_LIMIT_PER_MINUTE = 100
        mock_cache.get.return_value = cached
        from bot.handlers.ask import ask_handler
        await ask_handler(update, _make_context(args=["leave", "policy"]))
    reply_text = update.message.reply_text.call_args[0][0]
    assert "Cached answer" in reply_text
    assert "cached" in reply_text


# --- summarize handler ---

async def test_summarize_handler_no_history():
    update = _make_update(123)
    from bot.handlers import summarize
    with (
        patch("bot.middleware.allowlist.settings") as s1,
        patch("bot.middleware.rate_limit.settings") as s2,
        patch.object(summarize, "_conversation_history") as mock_hist,
    ):
        s1.ALLOWED_USER_IDS = [123]
        s2.RATE_LIMIT_PER_MINUTE = 100
        mock_hist.get_all_for_summary.return_value = []
        from bot.handlers.summarize import summarize_handler
        await summarize_handler(update, _make_context())
    update.message.reply_text.assert_called_once_with("No conversation history yet.")


# --- image handler ---

async def test_image_handler_no_photo_replies_with_prompt():
    update = _make_update(123)
    update.message.photo = []  # empty list = no photo
    from bot.handlers import image as image_mod
    with (
        patch("bot.middleware.allowlist.settings") as s1,
        patch("bot.middleware.rate_limit.settings") as s2,
    ):
        s1.ALLOWED_USER_IDS = [123]
        s2.RATE_LIMIT_PER_MINUTE = 100
        from bot.handlers.image import image_handler
        await image_handler(update, _make_context())
    update.message.reply_text.assert_called_once()
    assert "photo" in update.message.reply_text.call_args[0][0].lower()


async def test_image_handler_describes_photo():
    update = _make_update(123)
    photo = MagicMock(spec=PhotoSize)
    photo.file_id = "test_file_id"
    photo.file_size = 1024  # 1KB, well within limit
    update.message.photo = [photo]

    ctx = _make_context()
    mock_file = AsyncMock()
    mock_file.download_as_bytearray = AsyncMock(return_value=bytearray(b"fake_image_data"))
    ctx.bot = AsyncMock()
    ctx.bot.get_file = AsyncMock(return_value=mock_file)

    vision_result = VisionResult(caption="A cat on a mat", tags=["cat", "mat", "indoor"])

    from bot.handlers import image as image_mod
    with (
        patch("bot.middleware.allowlist.settings") as s1,
        patch("bot.middleware.rate_limit.settings") as s2,
        patch.object(image_mod, "_captioner") as mock_captioner,
        patch.object(image_mod, "_conversation_history") as mock_hist,
        patch.object(image_mod, "_db"),
    ):
        s1.ALLOWED_USER_IDS = [123]
        s2.RATE_LIMIT_PER_MINUTE = 100
        mock_captioner.describe = AsyncMock(return_value=vision_result)
        mock_hist.add = MagicMock()
        from bot.handlers.image import image_handler
        await image_handler(update, ctx)

    reply_text = update.message.reply_text.call_args[0][0]
    assert "A cat on a mat" in reply_text
    assert "cat" in reply_text
