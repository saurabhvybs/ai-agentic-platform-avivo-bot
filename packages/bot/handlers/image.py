import asyncio

import openai as openai_lib
from telegram import Update
from telegram.ext import ContextTypes
from telegram.helpers import escape_markdown

from bot.middleware.allowlist import allowed
from bot.middleware.rate_limit import rate_limited
from shared.config import settings
from shared.logger import logger, user_id_context
from storage.db import DBManager
from storage.history import ConversationHistory
from vision.captioner import ImageCaptioner

_db = DBManager.get_instance(settings.DB_PATH)
_conversation_history = ConversationHistory(_db, settings.MAX_HISTORY)
_openai_client = openai_lib.AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
_captioner = ImageCaptioner(_openai_client, settings.MAX_IMAGE_SIZE_MB)


@allowed
@rate_limited
async def image_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    user_id = update.effective_user.id
    user_id_context.set(str(user_id))

    if not update.message.photo:
        await update.message.reply_text("Please send a photo to get a description.")
        return

    # Highest resolution is last in the list
    photo = update.message.photo[-1]

    # Check size before downloading — file_size is available from Telegram metadata
    file_size_mb = (photo.file_size or 0) / (1024 * 1024)
    if file_size_mb > settings.MAX_IMAGE_SIZE_MB:
        await update.message.reply_text(
            f"\u26a0\ufe0f Image is too large. Maximum size is {settings.MAX_IMAGE_SIZE_MB}MB."
        )
        return

    await update.message.chat.send_action("typing")

    try:
        file = await context.bot.get_file(photo.file_id)
        image_bytes = bytes(await file.download_as_bytearray())

        result = await _captioner.describe(image_bytes)

        await asyncio.to_thread(_conversation_history.add, user_id, "user", "[sent an image]")
        await asyncio.to_thread(
            _conversation_history.add,
            user_id,
            "assistant",
            f"Caption: {result.caption}\nTags: {', '.join(result.tags)}",
        )

        caption = escape_markdown(result.caption, version=2)
        tags = escape_markdown(", ".join(result.tags), version=2)
        reply = f"\U0001f5bc\ufe0f *Caption:*\n{caption}\n\n\U0001f3f7\ufe0f *Tags:* {tags}"
        await update.message.reply_text(reply, parse_mode="MarkdownV2")

    except openai_lib.BadRequestError as e:
        logger.error(f"OpenAI vision rejected image: {e}")
        await update.message.reply_text("\u26a0\ufe0f Could not analyze this image.")
    except Exception:
        logger.error("image_handler error", exc_info=True)
        await update.message.reply_text("\u26a0\ufe0f Could not process the image. Please try again.")
