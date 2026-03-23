# Avivo Bot — Hybrid Telegram AI Bot

A production-grade Telegram bot that answers HR policy questions using Retrieval-Augmented Generation (RAG), describes images via OpenAI Vision, and optionally augments answers with live web search.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Telegram User                               │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                    ┌───────────▼───────────┐
                    │  AllowlistMiddleware  │──► "Not authorized"
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │  RateLimitMiddleware  │──► "Slow down!"
                    └───────────┬───────────┘
                                │
          ┌─────────────────────┼──────────────────────┐
          │                     │                      │
    ┌─────▼──────┐      ┌───────▼────────┐     ┌──────▼───────┐
    │  /ask      │      │  Photo Message │     │  /summarize  │
    │  Command   │      │  Handler       │     │  Command     │
    └─────┬──────┘      └───────┬────────┘     └──────┬───────┘
          │                     │                      │
    ┌─────▼──────────┐   ┌──────▼─────────┐    ┌──────▼──────────┐
    │  RAG Pipeline  │   │ Vision Pipeline│    │ History Summary │
    │                │   │                │    │                 │
    │ 1. Cache check │   │ 1. Size check  │    │ 1. Fetch last   │
    │ 2. Embed query │   │ 2. Download    │    │    50 messages  │
    │ 3. KNN search  │   │ 3. GPT-4o-mini │    │ 2. GPT-4o-mini  │
    │ 4. LLM answer  │   │    vision API  │    │    summarize    │
    │ 5. Web search? │   │ 4. Tags+caption│    │ 3. Reply        │
    │ 6. Cache+store │   │ 5. Store hist  │    └─────────────────┘
    └─────┬──────────┘   └──────┬─────────┘
          │                     │
    ┌─────▼─────────────────────▼─────────────────────┐
    │                    Storage Layer                 │
    │                                                  │
    │   SQLite DB (.db/genai_bot.db)                   │
    │   ├── document_chunks   (ingested text)          │
    │   ├── chunk_embeddings  (sqlite-vec KNN index)   │
    │   ├── query_cache       (TTL-based answer cache) │
    │   └── conversation_history (per-user messages)  │
    └──────────────────────────────────────────────────┘
```

### Ingest Flow (one-time / on update)

```
data/knowledge_base/*.md
        │
   ┌────▼──────┐     ┌──────────┐     ┌──────────────┐     ┌─────────────┐
   │  Loader   │────►│  Chunker │────►│   Embedder   │────►│ VectorStore │
   │ reads .md │     │ 400-char │     │ all-MiniLM   │     │ sqlite-vec  │
   │  files    │     │ overlaps │     │ L6-v2 (local)│     │ KNN index   │
   └───────────┘     └──────────┘     └──────────────┘     └─────────────┘
```

---

## Bot

**File:** `packages/bot/`

The bot is built on `python-telegram-bot 21+` in async mode. Every handler is wrapped with two decorator middlewares — `@allowed` then `@rate_limited` — applied in that order so unauthorized users are rejected before any rate-limit accounting.

| Handler | Trigger | Key behaviour |
|---|---|---|
| `ask_handler` | `/ask <query>` | Cache-first RAG → reply with MarkdownV2 |
| `image_handler` | Photo message | Vision analysis → caption + tags |
| `summarize_handler` | `/summarize` | Summarize last 50 history entries |
| `help_handler` | `/help` | Static usage text |

**Middleware:**
- **Allowlist** — `ALLOWED_USER_IDS` in `.env`. Empty = open bot (warned at startup). Non-empty = strict allowlist.
- **Rate limiter** — Sliding 60-second window per user. Default 10 req/min, configurable via `RATE_LIMIT_PER_MINUTE`.

**Graceful shutdown** — `post_stop` hook closes both RAG generators, the vision OpenAI client, and the SQLite connection in isolated try/except blocks so one failure does not skip the rest.

---

## RAG Pipeline

**Files:** `packages/rag/`

Answers `/ask` queries against the ingested knowledge base.

```
Query
  │
  ├─► QueryCache.get()          — MD5-keyed, TTL-checked; expired rows deleted on read
  │       hit ─► reply instantly (marked ⚡ cached)
  │       miss ─► continue
  │
  ├─► TextEmbedder.embed()      — all-MiniLM-L6-v2 via sentence-transformers (local, 384-dim)
  │
  ├─► RAGSearcher.retrieve()    — sqlite-vec KNN search, returns top-K (text, doc_name) pairs
  │
  ├─► RAGGenerator.generate()   — builds prompt from chunks + conversation history,
  │       primary: OpenAI gpt-4o-mini
  │       fallback: Ollama (if USE_OLLAMA=true)
  │
  ├─► Tavily web search          — appended if ENABLE_WEB_SEARCH=true and answer is weak
  │
  └─► QueryCache.set()          — store result; ConversationHistory.add() for user + assistant
```

All embedding and SQLite calls are wrapped in `asyncio.to_thread()` to keep the event loop non-blocking.

---

## Vision Pipeline

**File:** `packages/vision/captioner.py`

Handles photo messages sent directly to the bot (no command needed). Documents are intentionally excluded — PHOTO only.

**Flow:**
1. Size guard — rejects images over `MAX_IMAGE_SIZE_MB` (default 20 MB) before downloading
2. Downloads the highest-resolution version Telegram provides
3. Sends the image as a base64 data URL to `gpt-4o-mini` vision with a structured prompt
4. Parses response into a `VisionResult` — `caption` (1–2 sentences) + `tags` (list of keywords)
5. Stores caption+tags in conversation history for context
6. Replies with MarkdownV2-escaped output

---

## Storage

**Files:** `packages/storage/`

Single SQLite database file (`.db/genai_bot.db`) with the `sqlite-vec` extension for vector search.

| Module | Table | Purpose |
|---|---|---|
| `db.py` | — | Singleton connection; WAL mode; loads sqlite-vec extension |
| `vector_store.py` | `document_chunks`, `chunk_embeddings` | Insert/search/delete chunks; KNN via sqlite-vec |
| `cache.py` | `query_cache` | TTL cache for RAG answers; expired rows deleted on read |
| `history.py` | `conversation_history` | Per-user message log; trimmed to last `MAX_HISTORY` on write |

**Design decisions:**
- **Single file DB** — no external database needed; simple to back up and migrate
- **WAL mode** — allows concurrent reads during writes (ingest + bot running simultaneously)
- **Atomic deletes** — `vector_store.delete_by_doc()` uses `with conn:` for transactional safety
- **Singleton DBManager** — one connection shared across all handlers; serializes all writes, no race conditions

---

## Ingest

**Files:** `scripts/ingest.py`, `packages/rag/ingestion/`

One-time (or on-update) pipeline to load documents into the knowledge base.

```bash
# Docker
docker compose --profile ingest up

# Local
python scripts/ingest.py [--force]   # --force re-ingests and clears cache
```

**Steps:**
1. `Loader` — reads all `.md` files from `data/knowledge_base/`
2. `Chunker` — splits text into 400-char chunks with 50-char overlap (NLTK sentence-aware)
3. `Embedder` — embeds each chunk with `all-MiniLM-L6-v2` in batches of 32
4. `VectorStore.insert()` — stores text + sqlite-vec embedding; skips already-ingested docs unless `--force`

**Idempotent** — re-running without `--force` skips existing documents. With `--force`, the old document's chunks and embeddings are deleted atomically before re-ingesting, and the query cache is cleared.

---

## Setup & Run

### Local

```bash
pip install -e ".[dev]"
cp .env.example .env
# Set TELEGRAM_BOT_TOKEN, OPENAI_API_KEY, ALLOWED_USER_IDS

python scripts/ingest.py          # one-time knowledge base ingest
python -m packages.bot.main       # run bot
```

### Docker

```bash
docker compose --profile ingest up   # ingest docs (exits when done)
docker compose up -d bot             # start bot in background

# Optional: local Ollama LLM
docker compose --profile ollama up -d
# Set OLLAMA_BASE_URL=http://ollama:11434 in .env
```

---

## Configuration

```ini
# Required
TELEGRAM_BOT_TOKEN=...          # from @BotFather
OPENAI_API_KEY=...              # required for RAG + Vision

# Access control
ALLOWED_USER_IDS=123,456        # leave empty = open bot (warning logged)

# LLM
OPENAI_MODEL=gpt-4o-mini
USE_OLLAMA=false                # set true to use local Ollama instead
OLLAMA_MODEL=llama3.2

# RAG tuning
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHUNK_SIZE=400
CHUNK_OVERLAP=50
TOP_K=3                         # number of chunks retrieved per query
MAX_HISTORY=10                  # conversation turns sent as LLM context

# Web search
ENABLE_WEB_SEARCH=false
TAVILY_API_KEY=                 # required only if ENABLE_WEB_SEARCH=true

# Limits
RATE_LIMIT_PER_MINUTE=10
MAX_IMAGE_SIZE_MB=20
CACHE_TTL_HOURS=24
```

---

## Bot Commands

| Command | Description |
|---|---|
| `/ask <question>` | Query the knowledge base |
| `/summarize` | Summarize your conversation history |
| `/help` | Show available commands |
| Send a photo | Analyse the image (caption + tags) |

---

## Tests

```bash
pytest -v                           # unit tests (no API key needed)
OPENAI_API_KEY=sk-... pytest -v     # includes integration test
```

80+ unit tests covering all layers: storage, RAG pipeline, vision, bot handlers, middleware. Integration test runs a full embed → ingest → query → answer cycle against a real temporary database.
