import base64

import openai

from shared.models import VisionResult

VISION_PROMPT = (
    "Describe this image in 2-3 sentences. "
    "Then provide exactly 3 keyword tags as a comma-separated list prefixed with 'Tags:'"
)


class ImageCaptioner:
    def __init__(self, client: openai.AsyncOpenAI, max_size_mb: int = 20) -> None:
        self._client = client
        # Stored for handler introspection; size enforcement occurs before download
        # in the Telegram image handler, not here.
        self._max_size_mb = max_size_mb

    async def describe(self, image_bytes: bytes) -> VisionResult:
        """Describe image using OpenAI gpt-4o-mini vision.

        Size check is done in the handler (reads Telegram file_size before download).
        This method trusts that image_bytes is within limits.
        """
        base64_image = base64.b64encode(image_bytes).decode()
        response = await self._client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                        {"type": "text", "text": VISION_PROMPT},
                    ],
                }
            ],
            max_tokens=300,
        )
        content = response.choices[0].message.content or ""

        if "Tags:" in content:
            parts = content.split("Tags:", 1)
            caption = parts[0].strip()
            tags_str = parts[1].strip()
            tags = [t.strip() for t in tags_str.split(",")][:3]
        else:
            caption = content.strip()
            tags = []

        return VisionResult(caption=caption, tags=tags)
