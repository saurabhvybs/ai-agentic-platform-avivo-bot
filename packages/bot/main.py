import logging

from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters

from shared.config import settings
from shared.logger import configure_logging
from bot.handlers.ask import ask_handler, _rag_generator
from bot.handlers.image import image_handler
from bot.handlers.help import help_handler
from bot.handlers.summarize import summarize_handler
from storage.db import DBManager
from storage.vector_store import VectorStore


async def _on_shutdown(app) -> None:
    """Graceful shutdown: close all async clients and SQLite connection.

    Each close is wrapped individually so a failure in one does not skip the rest.
    """
    _log = logging.getLogger("avivo_bot")
    from bot.handlers.image import _openai_client as _vision_client
    from bot.handlers.summarize import _rag_generator as _summarize_rag_generator

    for close_fn, label in [
        (_rag_generator.close, "ask RAGGenerator"),
        (_summarize_rag_generator.close, "summarize RAGGenerator"),
        (_vision_client.close, "vision OpenAI client"),
    ]:
        try:
            await close_fn()
        except Exception:
            _log.warning("Error closing %s during shutdown", label, exc_info=True)

    try:
        DBManager.get_instance(settings.DB_PATH).close()
    except Exception:
        _log.warning("Error closing DBManager during shutdown", exc_info=True)

    _log.info("Shutdown complete.")


def main() -> None:
    configure_logging(settings.LOG_LEVEL)
    _log = logging.getLogger("avivo_bot")

    if not settings.ALLOWED_USER_IDS:
        _log.warning("ALLOWED_USER_IDS not set — bot is open to all users")

    if not VectorStore(DBManager.get_instance(settings.DB_PATH)).has_doc_any():
        _log.warning("Knowledge base is empty — run: docker compose --profile ingest up")

    app = ApplicationBuilder().token(settings.TELEGRAM_BOT_TOKEN).build()
    app.post_stop = _on_shutdown

    app.add_handler(CommandHandler("ask", ask_handler))
    app.add_handler(CommandHandler("help", help_handler))
    app.add_handler(CommandHandler("summarize", summarize_handler))
    # PHOTO only — Documents are excluded (intentional)
    app.add_handler(MessageHandler(filters.PHOTO, image_handler))

    logging.getLogger("avivo_bot").info("Bot starting...")
    app.run_polling()


if __name__ == "__main__":
    main()
