# Avivo Bot — Hybrid Telegram Bot (RAG + Vision)

A production-grade Telegram bot that answers HR policy questions via RAG and describes uploaded images via OpenAI Vision.

## Architecture

```
User (Telegram)
    │
[AllowlistMiddleware] → blocked → "⛔ Access denied"
[RateLimitMiddleware] → exceeded → "⚠️ Slow down"
    │
/ask → RAG Pipeline (sqlite-vec + OpenAI/Ollama + optional Tavily)
photo → Vision Pipeline (OpenAI gpt-4o-mini vision)
/summarize → History summarizer
```

## Tech Stack

| Component | Technology |
|---|---|
| Bot | python-telegram-bot 21+ |
| Embeddings | all-MiniLM-L6-v2 (local) |
| Vector store | sqlite-vec (single file) |
| RAG LLM | OpenAI gpt-4o-mini (primary) / Ollama (fallback) |
| Vision | OpenAI gpt-4o-mini vision |
| Web search | Tavily (optional) |

## Setup & Run Locally

```bash
# 1. Install dependencies
pip install -e ".[dev]"

# 2. Configure environment
cp .env.example .env
# Edit .env: set TELEGRAM_BOT_TOKEN, OPENAI_API_KEY, ALLOWED_USER_IDS

# 3. Ingest knowledge base
python scripts/ingest.py

# 4. Run the bot
python -m packages.bot.main
```

## Docker Compose

```bash
# One-time ingestion
docker compose --profile ingest up

# Start bot
docker compose up -d bot

# Optional: Ollama (if USE_OLLAMA=true)
docker compose --profile ollama up -d
# Set OLLAMA_BASE_URL=http://ollama:11434 in .env when running in Docker
```

## Bot Commands

| Command | Description | Example |
|---|---|---|
| `/ask <query>` | Ask a policy question | `/ask what is the leave policy?` |
| `[send photo]` | Describe any uploaded image | (just send a photo) |
| `/summarize` | Summarize conversation history | `/summarize` |
| `/help` | Show usage instructions | `/help` |

## Configuration

Key `.env` variables:

```ini
TELEGRAM_BOT_TOKEN=...      # From @BotFather
OPENAI_API_KEY=...          # Required (Vision always uses OpenAI)
ALLOWED_USER_IDS=123,456    # Comma-separated Telegram user IDs
USE_OLLAMA=false            # Set true for local LLM (RAG only)
ENABLE_WEB_SEARCH=false     # Set true + TAVILY_API_KEY for web augmentation
```

## Running Tests

```bash
pytest -v                          # Unit tests (no API key needed)
OPENAI_API_KEY=sk-... pytest -v    # Includes integration test
```
