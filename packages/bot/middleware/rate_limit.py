import time
from collections import deque
from functools import wraps

from telegram import Update
from telegram.ext import ContextTypes

from shared.config import settings

_user_timestamps: dict[int, deque] = {}


def rate_limited(func):
    """Decorator: sliding window rate limiter per user_id."""
    @wraps(func)
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not update.effective_user:
            return
        user_id = update.effective_user.id
        now = time.monotonic()
        timestamps = _user_timestamps.setdefault(user_id, deque())
        # Pop entries older than 60 seconds (sliding window)
        while timestamps and now - timestamps[0] > 60:
            timestamps.popleft()
        if len(timestamps) >= settings.RATE_LIMIT_PER_MINUTE:
            if update.message:
                await update.message.reply_text(
                    "Slow down! Please wait before sending another request."
                )
            return
        timestamps.append(now)
        return await func(update, context)
    return wrapper
