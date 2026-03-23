import asyncio

from telegram import Update
from telegram.ext import ContextTypes
from telegram.helpers import escape_markdown

from bot.middleware.allowlist import allowed
from bot.middleware.rate_limit import rate_limited
from shared.config import settings
from shared.logger import logger, user_id_context
from storage.db import DBManager
from storage.cache import QueryCache
from storage.history import ConversationHistory
from storage.vector_store import VectorStore
from rag.ingestion.embedder import TextEmbedder
from rag.retrieval.searcher import RAGSearcher
from rag.generation.generator import RAGGenerator

_db = DBManager.get_instance(settings.DB_PATH)
_vector_store = VectorStore(_db)
_query_cache = QueryCache(_db, settings.CACHE_TTL_HOURS)
_conversation_history = ConversationHistory(_db, settings.MAX_HISTORY)
_embedder = TextEmbedder(settings.EMBEDDING_MODEL)
_searcher = RAGSearcher(_embedder, _vector_store, settings.TOP_K)
_rag_generator = RAGGenerator(settings)


def _format_reply(result) -> str:
    answer = escape_markdown(result.answer, version=2)
    lines = [f"\U0001f4c4 *Answer:*\n{answer}"]
    if result.sources:
        sources = escape_markdown(', '.join(result.sources), version=2)
        lines.append(f"\n\U0001f4da *Sources:* {sources}")
    if result.web_references:
        web_parts = []
        for r in result.web_references:
            title = escape_markdown(r.get('title', 'Link'), version=2)
            url = escape_markdown(r['url'], version=2)
            web_parts.append(f"[{title}]({url})")
        links = ", ".join(web_parts)
        lines.append(f"\n\U0001f310 *Web:* {links}")
    return "\n".join(lines)


@allowed
@rate_limited
async def ask_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    user_id_context.set(str(user_id))

    query = " ".join(context.args) if context.args else ""
    if not query.strip():
        await update.message.reply_text(
            "Please provide a query: `/ask what is the leave policy?`",
            parse_mode="MarkdownV2",
        )
        return

    await update.message.chat.send_action("typing")

    cached = await asyncio.to_thread(_query_cache.get, query)
    if cached:
        # Cached results are not added to conversation history to avoid
        # duplicate entries when the same query is repeated.
        await update.message.reply_text(
            _format_reply(cached) + "\n\n\u26a1 _(cached)_", parse_mode="MarkdownV2"
        )
        return

    try:
        history = await asyncio.to_thread(_conversation_history.get, user_id)
        chunks = await _searcher.retrieve(query)
        result = await _rag_generator.generate(
            query, chunks, history, enable_web_search=settings.ENABLE_WEB_SEARCH
        )
        await asyncio.to_thread(_query_cache.set, query, result)
        await asyncio.to_thread(_conversation_history.add, user_id, "user", query)
        await asyncio.to_thread(
            _conversation_history.add, user_id, "assistant", result.answer
        )
        await update.message.reply_text(_format_reply(result), parse_mode="MarkdownV2")
    except Exception:
        logger.error("ask_handler error", exc_info=True)
        await update.message.reply_text("\u26a0\ufe0f Something went wrong. Please try again.")
