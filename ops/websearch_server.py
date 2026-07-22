"""Brave Search browser-scrape sidecar.

Runs an isolated headless Chromium (Playwright, ``--headless=new`` — no virtual
display needed) and scrapes Brave Search result pages. Brave gives Google-tier
results, often a direct featured answer, is keyless, and is not a Google
dependency. Kept in its own service so the browser can never destabilise the
locked-down voice orchestrator; the orchestrator calls it over loopback and
falls back to the keyless ``ddgs`` backend if it is unavailable.

    POST /search  {"query": str, "maxResults": int}
                  -> {"backend":"brave","query","answer","results":[{title,url,snippet}],"sources"}
    GET  /health  {"ok": true, "browserReady": bool}

The browser is launched lazily on the first query, reused across queries, and
closed after an idle period to release memory.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from urllib.parse import quote_plus, urlparse

import uvicorn
from fastapi import FastAPI
from playwright.async_api import Browser, Playwright, async_playwright
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("websearch-server")

_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/149.0.0.0 Safari/537.36"
)
_LAUNCH_ARGS = [
    "--headless=new",
    "--no-sandbox",
    "--disable-dev-shm-usage",
    "--disable-gpu",
    "--disable-blink-features=AutomationControlled",
]
_IDLE_CLOSE_SECONDS = float(os.environ.get("WEBSEARCH_IDLE_CLOSE_SECONDS", "600"))
_NAV_TIMEOUT_MS = int(os.environ.get("WEBSEARCH_NAV_TIMEOUT_MS", "20000"))
_SETTLE_MS = int(os.environ.get("WEBSEARCH_SETTLE_MS", "2200"))

# Defensive DOM extraction — Brave's class names drift, so fall back to whole
# snippet innerText. Returns a featured answer (when Brave shows one) plus the
# top result blocks.
_EXTRACT_JS = r"""
(max) => {
  const norm = s => (s || '').replace(/\s+/g, ' ').trim();
  const ansEl = document.querySelector(
    '[data-type="infobox"] .desc, .answer, #infobox .desc,'
    + ' [class*="answer"] .desc, [class*="answer"]');
  const answer = ansEl ? norm(ansEl.innerText).slice(0, 500) : '';
  const out = [];
  const blocks = document.querySelectorAll('#results .snippet, main .snippet, [data-type="web"]');
  for (const b of blocks) {
    if (out.length >= max) break;
    const a = b.querySelector('a[href^="http"]');
    if (!a) continue;
    const title = norm(b.querySelector('.title, .h, [class*="title"]')?.innerText || a.innerText);
    const desc = norm(
      b.querySelector('.snippet-description, .snippet-content, [class*="desc"]')?.innerText
      || b.innerText);
    if (desc) out.push({ title: title.slice(0, 140), url: a.href, snippet: desc.slice(0, 320) });
  }
  return { answer, results: out };
}
"""


def _domain(url: str) -> str | None:
    try:
        host = urlparse(url).hostname
    except ValueError:
        return None
    if not host:
        return None
    return host[4:] if host.startswith("www.") else host


class _BrowserManager:
    """A single lazily-launched, reused, idle-closing Chromium."""

    def __init__(self) -> None:
        self._pw: Playwright | None = None
        self._browser: Browser | None = None
        self._page = None
        self._lock = asyncio.Lock()
        self._last_used = 0.0

    @property
    def ready(self) -> bool:
        return self._browser is not None and self._browser.is_connected()

    async def _ensure(self) -> None:
        if self.ready and self._page is not None and not self._page.is_closed():
            return
        await self._teardown()
        self._pw = await async_playwright().start()
        self._browser = await self._pw.chromium.launch(headless=False, args=_LAUNCH_ARGS)
        context = await self._browser.new_context(
            locale="en-NZ", user_agent=_UA, viewport={"width": 1280, "height": 900}
        )
        await context.add_init_script(
            "Object.defineProperty(navigator,'webdriver',{get:()=>undefined})"
        )
        self._page = await context.new_page()

    async def _teardown(self) -> None:
        try:
            if self._browser is not None:
                await self._browser.close()
        except Exception:  # best-effort
            logger.warning("browser close failed", exc_info=True)
        finally:
            self._browser = None
            self._page = None
        try:
            if self._pw is not None:
                await self._pw.stop()
        except Exception:
            logger.warning("playwright stop failed", exc_info=True)
        finally:
            self._pw = None

    async def search(self, query: str, max_results: int) -> dict:
        async with self._lock:
            await self._ensure()
            assert self._page is not None
            url = "https://search.brave.com/search?q=" + quote_plus(query)
            await self._page.goto(url, wait_until="domcontentloaded", timeout=_NAV_TIMEOUT_MS)
            await self._page.wait_for_timeout(_SETTLE_MS)
            data = await self._page.evaluate(_EXTRACT_JS, max_results)
            self._last_used = time.monotonic()
        results = data.get("results") or []
        sources: list[str] = []
        for item in results:
            domain = _domain(str(item.get("url") or ""))
            if domain and domain not in sources:
                sources.append(domain)
        return {
            "backend": "brave",
            "query": query,
            "answer": str(data.get("answer") or ""),
            "results": results,
            "sources": sources[:5],
        }

    async def idle_reaper(self) -> None:
        while True:
            await asyncio.sleep(60)
            if self.ready and self._last_used and (
                time.monotonic() - self._last_used > _IDLE_CLOSE_SECONDS
            ):
                async with self._lock:
                    if self._last_used and (
                        time.monotonic() - self._last_used > _IDLE_CLOSE_SECONDS
                    ):
                        logger.info("closing idle browser after %.0fs", _IDLE_CLOSE_SECONDS)
                        await self._teardown()

    async def close(self) -> None:
        async with self._lock:
            await self._teardown()


_manager = _BrowserManager()


@asynccontextmanager
async def _lifespan(app: FastAPI):
    reaper = asyncio.create_task(_manager.idle_reaper())
    try:
        yield
    finally:
        reaper.cancel()
        await _manager.close()


app = FastAPI(title="Nova Voice Brave Search sidecar", lifespan=_lifespan)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=400)
    maxResults: int = Field(default=4, ge=1, le=8)


@app.get("/health")
async def health() -> dict:
    return {"ok": True, "backend": "brave", "browserReady": _manager.ready}


@app.post("/search")
async def search(request: SearchRequest) -> dict:
    from fastapi import HTTPException

    try:
        result = await _manager.search(request.query.strip(), request.maxResults)
    except Exception as error:  # scrape/browser failure -> caller falls back
        logger.warning("brave search failed: %s", error, exc_info=True)
        raise HTTPException(status_code=502, detail="brave search unavailable") from error
    if not result["answer"] and not result["results"]:
        raise HTTPException(status_code=502, detail="no brave results")
    return result


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("WEBSEARCH_PORT", "8093")))
