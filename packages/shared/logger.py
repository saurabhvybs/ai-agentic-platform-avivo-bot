import logging
from contextvars import ContextVar

user_id_context: ContextVar[str] = ContextVar("user_id", default="system")

logger = logging.getLogger("avivo_bot")


class UserIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.user_id = user_id_context.get()
        return True


def configure_logging(level: str = "INFO") -> None:
    """Call once in bot/main.py before handler registration."""
    handler = logging.StreamHandler()
    handler.addFilter(UserIdFilter())
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [user:%(user_id)s] %(name)s: %(message)s"
    )
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
