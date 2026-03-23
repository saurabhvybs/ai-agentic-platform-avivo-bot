from telegram import Update
from telegram.ext import ContextTypes

from bot.middleware.allowlist import allowed
from bot.middleware.rate_limit import rate_limited

HELP_TEXT = (
    "\U0001f916 Avivo Bot \u2014 Commands\n"
    "\n"
    "/ask <query>  \u2014 Ask a question from company policies\n"
    "  Example: /ask what is the remote work policy?\n"
    "\n"
    "\U0001f4f7 Send any photo  \u2014 Get an image description and tags\n"
    "\n"
    "/summarize  \u2014 Summarize your last conversation\n"
    "\n"
    "/help  \u2014 Show this message"
)


@allowed
@rate_limited
async def help_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(HELP_TEXT)
