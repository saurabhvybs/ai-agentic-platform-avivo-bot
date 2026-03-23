import pytest
import time
from collections import deque
from unittest.mock import AsyncMock, MagicMock, patch
from telegram import Update, User, Message
from bot.middleware import rate_limit as rl_module
from bot.middleware.rate_limit import rate_limited


def _make_update(user_id: int):
    user = MagicMock(spec=User)
    user.id = user_id
    message = MagicMock(spec=Message)
    message.reply_text = AsyncMock()
    update = MagicMock(spec=Update)
    update.effective_user = user
    update.message = message
    return update


@pytest.fixture(autouse=True)
def clear_timestamps():
    rl_module._user_timestamps.clear()
    yield
    rl_module._user_timestamps.clear()


async def test_first_request_passes():
    update = _make_update(1)
    called = False

    with patch("bot.middleware.rate_limit.settings") as s:
        s.RATE_LIMIT_PER_MINUTE = 5

        @rate_limited
        async def handler(u, c):
            nonlocal called
            called = True

        await handler(update, MagicMock())

    assert called


async def test_under_limit_all_pass():
    update = _make_update(1)
    call_count = 0

    with patch("bot.middleware.rate_limit.settings") as s:
        s.RATE_LIMIT_PER_MINUTE = 3

        @rate_limited
        async def handler(u, c):
            nonlocal call_count
            call_count += 1

        for _ in range(3):
            await handler(update, MagicMock())

    assert call_count == 3


async def test_over_limit_blocks():
    update = _make_update(1)
    call_count = 0

    with patch("bot.middleware.rate_limit.settings") as s:
        s.RATE_LIMIT_PER_MINUTE = 2

        @rate_limited
        async def handler(u, c):
            nonlocal call_count
            call_count += 1

        for _ in range(4):
            await handler(update, MagicMock())

    assert call_count == 2
    assert update.message.reply_text.call_count == 2  # 2 blocked replies


async def test_sliding_window_resets_old_entries():
    update = _make_update(1)
    call_count = 0

    with patch("bot.middleware.rate_limit.settings") as s:
        s.RATE_LIMIT_PER_MINUTE = 2

        @rate_limited
        async def handler(u, c):
            nonlocal call_count
            call_count += 1

        # Manually insert old timestamps (>60s ago)
        rl_module._user_timestamps[1] = deque([time.monotonic() - 61, time.monotonic() - 61])

        # Now two more requests should pass (old ones slide out)
        await handler(update, MagicMock())
        await handler(update, MagicMock())

    assert call_count == 2
