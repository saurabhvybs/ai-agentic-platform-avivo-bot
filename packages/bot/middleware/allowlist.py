from functools import wraps

from telegram import Update
from telegram.ext import ContextTypes

from shared.config import settings
from shared.logger import logger


def allowed(func):
    """Decorator: checks user_id against ALLOWED_USER_IDS before calling handler."""
    @wraps(func)
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not update.effective_user:
            return
        user_id = update.effective_user.id
        if settings.ALLOWED_USER_IDS and user_id not in settings.ALLOWED_USER_IDS:
            logger.warning(f"Unauthorized access attempt by user_id={user_id}")
            if update.message:
                await update.message.reply_text("You are not authorized to use this bot.")
            return
        return await func(update, context)
    return wrapper
