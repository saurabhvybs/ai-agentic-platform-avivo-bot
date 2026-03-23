from pydantic import field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Telegram
    TELEGRAM_BOT_TOKEN: str = "placeholder"
    ALLOWED_USER_IDS: list[int] = []

    # OpenAI
    OPENAI_API_KEY: str = "placeholder"
    OPENAI_MODEL: str = "gpt-4o-mini"

    # Ollama fallback
    USE_OLLAMA: bool = False
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.2"

    # RAG
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    DB_PATH: str = ".db/genai_bot.db"
    CHUNK_SIZE: int = 400
    CHUNK_OVERLAP: int = 50
    TOP_K: int = 3
    MAX_HISTORY: int = 10

    # Web search
    TAVILY_API_KEY: str | None = None
    ENABLE_WEB_SEARCH: bool = False

    # Rate limiting
    RATE_LIMIT_PER_MINUTE: int = 10

    # Logging
    LOG_LEVEL: str = "INFO"

    # Vision
    MAX_IMAGE_SIZE_MB: int = 20

    # Cache
    CACHE_TTL_HOURS: int = 24

    @field_validator("ALLOWED_USER_IDS", mode="before")
    @classmethod
    def parse_allowed_user_ids(cls, v: str | list) -> list[int]:
        if isinstance(v, list):
            return [int(x) for x in v]
        return [int(x.strip()) for x in str(v).split(",") if x.strip()]

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
