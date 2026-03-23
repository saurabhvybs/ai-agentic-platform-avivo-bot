import asyncio

from telegram import Update
from telegram.ext import ContextTypes
from telegram.helpers import escape_markdown

from bot.middleware.allowlist import allowed
from bot.middleware.rate_limit import rate_limited
from shared.config import settings
from shared.logger import logger, user_id_context
from storage.db import DBManager
from storage.history import ConversationHistory
from rag.generation.generator import RAGGenerator

_db = DBManager.get_instance(settings.DB_PATH)
_conversation_history = ConversationHistory(_db, settings.MAX_HISTORY)
_rag_generator = RAGGenerator(settings)


@allowed
@rate_limited
async def summarize_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    user_id_context.set(str(user_id))

    # Use get_all_for_summary() — not get() — summary needs full conversation, not last-N
    history = await asyncio.to_thread(_conversation_history.get_all_for_summary, user_id)
    if not history:
        await update.message.reply_text("No conversation history yet.")
        return

    try:
        summary = await _rag_generator.summarize(history)
        summary_escaped = escape_markdown(summary, version=2)
        await update.message.reply_text(f"\U0001f4dd *Summary:*\n{summary_escaped}", parse_mode="MarkdownV2")
    except Exception:
        logger.error("summarize_handler error", exc_info=True)
        await update.message.reply_text("\u26a0\ufe0f Could not generate summary. Please try again.")
