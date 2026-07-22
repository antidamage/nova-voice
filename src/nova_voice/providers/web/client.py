"""Web backends for the voice agent's ``web.ask`` capability.

Two interchangeable, async backends answer a rewritten, self-contained query:

- :class:`GeminiClient` delegates to Google's free-tier grounded model (Search
  grounding) and returns a concise, text-to-speech-friendly answer plus the
  source domains it grounded on. **Only the query text is sent** — never
  persona, household state, or audio.
- :class:`WebSearchClient` is keyless: DuckDuckGo results plus a readable-text
  fetch of the top page, returned as raw context for the local model to
  summarize during response rendering.

Both mirror ``providers/nova/client.py``'s httpx idiom. The DuckDuckGo and
BeautifulSoup imports are lazy so the Gemini path works even if the optional
local-fallback dependencies are absent.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any
from urllib.parse import urlparse

import httpx


class WebBackendError(RuntimeError):
    """A web backend could not produce an answer for the query."""


def _domain(url: str) -> str | None:
    try:
        host = urlparse(url).hostname
    except ValueError:
        return None
    if not host:
        return None
    return host[4:] if host.startswith("www.") else host


class GeminiClient:
    """Google Gemini free-tier grounded-answer backend (Search grounding).

    Sends only the rewritten query with a system instruction to answer plainly
    for speech. Returns ``{"answer", "sources", "backend"}``.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str,
        *,
        timeout_seconds: float = 8,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=timeout_seconds,
            transport=transport,
            headers={"User-Agent": "nova-voice/0.1"},
        )

    async def answer(self, query: str, *, max_sentences: int) -> dict[str, Any]:
        instruction = (
            "You answer a single question for a voice assistant that reads your "
            "reply aloud. Use web search to ground the answer in current, factual "
            "information. Reply in plain spoken sentences with no markdown, lists, "
            f"links, or inline citations, in at most {max_sentences} sentence"
            f"{'s' if max_sentences != 1 else ''}. Be direct; if the answer is "
            "uncertain or unavailable, say so briefly."
        )
        payload = {
            "system_instruction": {"parts": [{"text": instruction}]},
            "contents": [{"role": "user", "parts": [{"text": query}]}],
            # Google's grounding tool for Gemini 2.x. (Older 1.5 models used
            # "google_search_retrieval"; the model id is config-driven so this
            # can be revisited if the pinned model changes.)
            "tools": [{"google_search": {}}],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 60 * max_sentences + 80,
            },
        }
        try:
            response = await self._client.post(
                f"/models/{self.model}:generateContent",
                headers={"x-goog-api-key": self.api_key},
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
        except (httpx.HTTPError, ValueError) as error:
            raise WebBackendError(f"gemini request failed: {error}") from error

        answer = _gemini_answer_text(data)
        if not answer:
            raise WebBackendError("gemini returned no answer text")
        return {
            "backend": "gemini",
            "answer": answer,
            "sources": _gemini_sources(data),
        }

    async def close(self) -> None:
        await self._client.aclose()


def _gemini_answer_text(data: dict[str, Any]) -> str:
    candidates = data.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return ""
    content = candidates[0].get("content") if isinstance(candidates[0], dict) else None
    parts = content.get("parts") if isinstance(content, dict) else None
    if not isinstance(parts, list):
        return ""
    texts = [
        part["text"]
        for part in parts
        if isinstance(part, dict) and isinstance(part.get("text"), str)
    ]
    return " ".join(text.strip() for text in texts if text.strip()).strip()


def _gemini_sources(data: dict[str, Any]) -> list[str]:
    candidates = data.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return []
    metadata = candidates[0].get("groundingMetadata") if isinstance(candidates[0], dict) else None
    chunks = metadata.get("groundingChunks") if isinstance(metadata, dict) else None
    if not isinstance(chunks, list):
        return []
    domains: list[str] = []
    for chunk in chunks:
        web = chunk.get("web") if isinstance(chunk, dict) else None
        uri = web.get("uri") if isinstance(web, dict) else None
        domain = _domain(uri) if isinstance(uri, str) else None
        if domain and domain not in domains:
            domains.append(domain)
    return domains[:5]


class BraveScrapeClient:
    """Client for the Brave Search browser-scrape sidecar (ops/websearch_server.py).

    The heavy browser work lives in an isolated service; this just POSTs the
    query over loopback and returns its ``{answer, results, sources}`` payload.
    """

    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float = 15,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=timeout_seconds,
            transport=transport,
            headers={"User-Agent": "nova-voice/0.1"},
        )

    async def gather(self, query: str, *, max_results: int = 4) -> dict[str, Any]:
        try:
            response = await self._client.post(
                "/search", json={"query": query, "maxResults": max_results}
            )
            response.raise_for_status()
            data = response.json()
        except (httpx.HTTPError, ValueError) as error:
            raise WebBackendError(f"brave sidecar request failed: {error}") from error
        if not isinstance(data, dict) or (not data.get("answer") and not data.get("results")):
            raise WebBackendError("brave sidecar returned no results")
        return {
            "backend": "brave",
            "query": query,
            "answer": str(data.get("answer") or ""),
            "results": data.get("results") or [],
            "sources": data.get("sources") or [],
        }

    async def close(self) -> None:
        await self._client.aclose()


class WebSearchClient:
    """Keyless DuckDuckGo search + readable-text fetch of the top result.

    Returns raw context (``results`` snippets + a page ``excerpt``) for the local
    model to summarize; it does not itself produce a finished answer.
    """

    def __init__(
        self,
        *,
        results: int = 4,
        fetch_max_bytes: int = 200_000,
        max_result_chars: int = 6000,
        timeout_seconds: float = 8,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.results = max(1, min(int(results), 10))
        self.fetch_max_bytes = max(4096, int(fetch_max_bytes))
        self.max_result_chars = max(500, int(max_result_chars))
        self._client = httpx.AsyncClient(
            timeout=timeout_seconds,
            transport=transport,
            follow_redirects=True,
            headers={"User-Agent": "nova-voice/0.1"},
        )

    async def gather(self, query: str) -> dict[str, Any]:
        results = await asyncio.to_thread(self._search, query)
        if not results:
            raise WebBackendError("no web results for query")
        excerpt = ""
        top_url = results[0].get("url") or ""
        if top_url:
            try:
                excerpt = await self._fetch(top_url)
            except (httpx.HTTPError, ValueError):
                excerpt = ""
        sources: list[str] = []
        for item in results:
            domain = _domain(str(item.get("url") or ""))
            if domain and domain not in sources:
                sources.append(domain)
        return {
            "backend": "local",
            "query": query,
            "results": results,
            "excerpt": excerpt[: self.max_result_chars],
            "sources": sources[:5],
        }

    def _search(self, query: str) -> list[dict[str, str]]:
        # Lazy import: the local backend is optional, so the module import never
        # hard-depends on it. Uses `ddgs` (the maintained successor to the
        # abandoned `duckduckgo-search`, whose scraper now returns wrong/region-
        # defaulted junk); same {title,href,body} result shape.
        from ddgs import DDGS

        with DDGS() as ddgs:
            raw = list(ddgs.text(query, max_results=self.results))
        return [
            {
                "title": str(item.get("title", "")),
                "url": str(item.get("href") or item.get("url", "")),
                "snippet": str(item.get("body", "")),
            }
            for item in raw
        ]

    async def _fetch(self, url: str) -> str:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError("web fetch only supports http and https URLs")
        raw = bytearray()
        async with self._client.stream("GET", url) as response:
            response.raise_for_status()
            content_type = response.headers.get("content-type", "")
            async for chunk in response.aiter_bytes():
                raw.extend(chunk)
                if len(raw) >= self.fetch_max_bytes:
                    break
        text = bytes(raw[: self.fetch_max_bytes]).decode("utf-8", errors="replace")
        if "html" in content_type or "<html" in text[:1000].lower():
            text = _readable_text(text)
        return text.strip()

    async def close(self) -> None:
        await self._client.aclose()


def _readable_text(html: str) -> str:
    # Lazy import for the same reason as the DuckDuckGo search above.
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text("\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text
