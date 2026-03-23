import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from rag.generation.generator import RAGGenerator
from shared.models import HistoryEntry, RAGResult
from datetime import datetime, timezone


def _make_settings(use_ollama=False, enable_web_search=False, tavily_key=None):
    s = MagicMock()
    s.USE_OLLAMA = use_ollama
    s.OPENAI_MODEL = "gpt-4o-mini"
    s.OPENAI_API_KEY = "test-key"
    s.OLLAMA_BASE_URL = "http://localhost:11434"
    s.OLLAMA_MODEL = "llama3.2"
    s.TAVILY_API_KEY = tavily_key
    s.ENABLE_WEB_SEARCH = enable_web_search
    return s


def _history_entry(role: str, content: str) -> HistoryEntry:
    return HistoryEntry(
        user_id=1, role=role, content=content, created_at=datetime.now(timezone.utc)
    )


@pytest.fixture
def generator():
    with patch("rag.generation.generator.openai.AsyncOpenAI"):
        with patch("rag.generation.generator.httpx.AsyncClient"):
            gen = RAGGenerator(_make_settings())
            return gen


async def test_generate_calls_openai_when_not_ollama(generator):
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "20 days annual leave."
    generator._openai.chat.completions.create = AsyncMock(return_value=mock_response)

    result = await generator.generate(
        query="What is leave?",
        retrieved_chunks=[("Annual leave is 20 days.", "leave_policy")],
        history=[],
    )
    assert isinstance(result, RAGResult)
    assert result.answer == "20 days annual leave."
    assert result.from_cache is False


async def test_generate_includes_history_in_messages(generator):
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Answer"
    generator._openai.chat.completions.create = AsyncMock(return_value=mock_response)

    history = [
        _history_entry("user", "previous question"),
        _history_entry("assistant", "previous answer"),
    ]
    await generator.generate("new query", [], history)

    call_args = generator._openai.chat.completions.create.call_args
    messages = call_args.kwargs["messages"]
    roles = [m["role"] for m in messages]
    assert "user" in roles
    assert "assistant" in roles


async def test_generate_includes_rag_context(generator):
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Answer"
    generator._openai.chat.completions.create = AsyncMock(return_value=mock_response)

    chunks = [("Leave is 20 days.", "leave_policy")]
    await generator.generate("query", chunks, [])

    call_args = generator._openai.chat.completions.create.call_args
    messages = call_args.kwargs["messages"]
    user_msg = next(m for m in messages if m["role"] == "user")
    assert "leave_policy" in user_msg["content"]
    assert "Leave is 20 days." in user_msg["content"]


async def test_generate_returns_sources(generator):
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Answer"
    generator._openai.chat.completions.create = AsyncMock(return_value=mock_response)

    chunks = [("text a", "doc_a"), ("text b", "doc_b")]
    result = await generator.generate("query", chunks, [])
    assert set(result.sources) == {"doc_a", "doc_b"}


async def test_generate_no_chunks_still_calls_llm(generator):
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "I don't know."
    generator._openai.chat.completions.create = AsyncMock(return_value=mock_response)

    result = await generator.generate("unknown query", [], [])
    assert result.answer == "I don't know."
    assert result.sources == []


async def test_generate_web_search_skipped_when_tavily_none(generator):
    """When self._tavily is None, web search is skipped regardless of flag."""
    generator._tavily = None
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Answer"
    generator._openai.chat.completions.create = AsyncMock(return_value=mock_response)

    result = await generator.generate("query", [], [], enable_web_search=True)
    assert result.web_references == []


async def test_generate_tavily_failure_returns_rag_only(generator):
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Answer"
    generator._openai.chat.completions.create = AsyncMock(return_value=mock_response)
    generator._tavily = MagicMock()
    generator._tavily.search.side_effect = Exception("Tavily down")

    result = await generator.generate("query", [], [], enable_web_search=True)
    assert result.web_references == []
    assert result.answer == "Answer"


async def test_summarize_returns_string(generator):
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "• Point 1\n• Point 2\n• Point 3"
    generator._openai.chat.completions.create = AsyncMock(return_value=mock_response)

    history = [_history_entry("user", "q"), _history_entry("assistant", "a")]
    summary = await generator.summarize(history)
    assert isinstance(summary, str)
    assert "Point 1" in summary


async def test_close_shuts_down_clients(generator):
    generator._http_client.aclose = AsyncMock()
    generator._openai.aclose = AsyncMock()
    await generator.close()
    generator._http_client.aclose.assert_called_once()
    generator._openai.aclose.assert_called_once()
