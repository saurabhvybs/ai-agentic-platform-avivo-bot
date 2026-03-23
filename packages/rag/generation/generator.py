import asyncio
from typing import Optional

import httpx
import openai
from tavily import TavilyClient

from shared.logger import logger
from shared.models import HistoryEntry, RAGResult

SYSTEM_PROMPT = (
    "You are a helpful HR assistant. Answer questions based on the provided context "
    "from company policy documents. If the answer is not in the context, say so clearly. "
    "Be concise and accurate."
)

SUMMARIZE_PROMPT = (
    "Summarize the following conversation in exactly 3 concise bullet points."
)


class RAGGenerator:
    def __init__(self, settings) -> None:
        self._use_ollama = settings.USE_OLLAMA
        self._openai_model = settings.OPENAI_MODEL
        self._ollama_base_url = settings.OLLAMA_BASE_URL
        self._ollama_model = settings.OLLAMA_MODEL
        self._openai = openai.AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        # httpx client for Ollama — created once, closed on shutdown
        self._http_client = httpx.AsyncClient()
        self._tavily: Optional[TavilyClient] = (
            TavilyClient(api_key=settings.TAVILY_API_KEY)
            if settings.TAVILY_API_KEY
            else None
        )

    async def _call_llm(self, messages: list[dict]) -> str:
        if self._use_ollama:
            resp = await self._http_client.post(
                f"{self._ollama_base_url}/api/chat",
                json={
                    "model": self._ollama_model,
                    "messages": messages,
                    "stream": False,
                },
                timeout=60.0,
            )
            resp.raise_for_status()
            return resp.json()["message"]["content"]
        else:
            response = await self._openai.chat.completions.create(
                model=self._openai_model,
                messages=messages,
            )
            return response.choices[0].message.content

    def _format_rag_context(self, chunks: list[tuple[str, str]]) -> str:
        if not chunks:
            return ""
        return "\n\n".join(f"[Source: {doc}]\n{text}" for text, doc in chunks)

    async def generate(
        self,
        query: str,
        retrieved_chunks: list[tuple[str, str]],
        history: list[HistoryEntry],
        enable_web_search: bool = False,
    ) -> RAGResult:
        rag_context = self._format_rag_context(retrieved_chunks)

        web_context, web_refs = "", []
        if enable_web_search and self._tavily:
            try:
                result = await asyncio.to_thread(
                    self._tavily.search, query, max_results=5
                )
                # SDK shape: {"results": [...], "answer": str|None}
                web_context = result.get("answer") or ""
                web_refs = [
                    {"url": r["url"], "title": r["title"]}
                    for r in result.get("results", [])
                ]
            except Exception as e:
                logger.warning(f"Tavily search failed: {e} — continuing RAG-only")

        context_parts = []
        if rag_context:
            context_parts.append(f"## Knowledge Base\n{rag_context}")
        if web_context:
            context_parts.append(f"## Web Results\n{web_context}")
        combined_context = "\n\n".join(context_parts)

        messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        for entry in history:
            messages.append({"role": entry.role, "content": entry.content})

        user_content = query
        if combined_context:
            user_content = f"{query}\n\nContext:\n{combined_context}"
        messages.append({"role": "user", "content": user_content})

        answer = await self._call_llm(messages)
        sources = list({doc for _, doc in retrieved_chunks})

        return RAGResult(
            answer=answer,
            sources=sources,
            web_references=web_refs,
            from_cache=False,
        )

    async def summarize(self, history: list[HistoryEntry]) -> str:
        """Summarize conversation in 3 bullet points. Uses same LLM as generate()."""
        messages: list[dict] = [{"role": "system", "content": SUMMARIZE_PROMPT}]
        for entry in history:
            messages.append({"role": entry.role, "content": entry.content})
        return await self._call_llm(messages)

    async def close(self) -> None:
        await self._http_client.aclose()
        await self._openai.aclose()
