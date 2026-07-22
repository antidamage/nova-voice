"""The ``web`` capability provider: a single read-only ``web.ask`` tool.

The local model rewrites a spoken request into a self-contained query and plans
one ``web.ask`` action. This provider runs the query against the configured
backend and returns the result as ``ToolResult.observed`` for the response
renderer to voice. It changes no household state, so its policy is low-risk and
it never mutates anything — the dashboard ``webAccessEnabled`` switch is the
consent gate, and the interpreter only offers the tool when that switch is on.
"""

from __future__ import annotations

import logging

from nova_voice.capabilities.base import CapabilityManifest, CapabilityProvider, ToolPolicy
from nova_voice.domain import PlannedAction, ToolResult
from nova_voice.providers.web.client import (
    BraveScrapeClient,
    GeminiClient,
    WebBackendError,
    WebSearchClient,
)

logger = logging.getLogger(__name__)


WEB_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web.ask",
            "description": (
                "Look something up on the web. Provide a single, self-contained "
                "search query rewritten from the user's request (include the "
                "who/what/where so it stands alone). Use only when the request "
                "needs current, real-world, or external facts you do not "
                "reliably know — news, live results, prices, today's events, or "
                "an unfamiliar topic. Never use it for household device control."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "minLength": 1, "maxLength": 400},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
]


class WebProvider(CapabilityProvider):
    def __init__(
        self,
        *,
        gemini: GeminiClient | None,
        search: WebSearchClient,
        brave: BraveScrapeClient | None = None,
        default_backend: str = "brave",
        answer_max_sentences: int = 2,
        search_results: int = 4,
        contract_version: str = "web-provider-v1",
    ) -> None:
        self._gemini = gemini
        self._search = search
        self._brave = brave
        self.search_results = max(1, int(search_results))
        self.contract_version = contract_version
        # Live-tunable from the dashboard via ``configure`` (mirrors
        # NovaProvider.configure_verification_loop).
        self.backend = self._normalize_backend(default_backend)
        self.answer_max_sentences = max(1, int(answer_max_sentences))

    @staticmethod
    def _normalize_backend(backend: str) -> str:
        return backend if backend in {"brave", "local", "gemini"} else "brave"

    def configure(self, *, backend: str, answer_max_sentences: int) -> None:
        """Apply live web settings collected from the dashboard."""

        self.backend = self._normalize_backend(backend)
        self.answer_max_sentences = max(1, int(answer_max_sentences))

    def manifest(self) -> CapabilityManifest:
        return CapabilityManifest(
            id="web",
            version="0.1.0",
            contract_version=self.contract_version,
            # Runs on iridium (the orchestrator makes the outbound call); it is
            # not a household LAN service like the dashboard.
            execution_class="iridium_local",
            tools=WEB_TOOLS,
            skill_files=[],
            tool_policies={
                # Read-only: retrieves information, mutates no household state.
                "web.ask": ToolPolicy(
                    risk="low",
                    reversible=True,
                    idempotent=True,
                    parallel_safe=True,
                    cancellation="anytime",
                ),
            },
        )

    async def execute(self, action: PlannedAction) -> ToolResult:
        if action.call.tool not in {"web.ask", "ask"}:
            return ToolResult(
                action_id=action.id,
                ok=False,
                code="blocked",
                message="Unknown web semantic tool",
            )
        query = str(action.call.arguments.get("query") or "").strip()
        if not query:
            return ToolResult(
                action_id=action.id,
                ok=False,
                code="invalid",
                message="Web lookup needs a query",
            )

        backend = self.backend
        # Resolve a selected-but-unavailable backend down to one that exists, so
        # the feature still answers rather than failing.
        if backend == "gemini" and self._gemini is None:
            backend = "brave" if self._brave is not None else "local"
        if backend == "brave" and self._brave is None:
            backend = "local"

        try:
            if backend == "gemini":
                assert self._gemini is not None
                try:
                    observed = await self._gemini.answer(
                        query, max_sentences=self.answer_max_sentences
                    )
                except WebBackendError:
                    logger.warning(
                        "gemini web backend failed; falling back to local", exc_info=True
                    )
                    observed = await self._search.gather(query)
            elif backend == "brave":
                assert self._brave is not None
                try:
                    observed = await self._brave.gather(query, max_results=self.search_results)
                except WebBackendError:
                    # Sidecar down or Brave blocked us: fall back to keyless ddgs.
                    logger.warning(
                        "brave web backend failed; falling back to local", exc_info=True
                    )
                    observed = await self._search.gather(query)
            else:
                observed = await self._search.gather(query)
        except WebBackendError as error:
            return ToolResult(
                action_id=action.id,
                ok=False,
                code="backend_error",
                target="web",
                requested={"query": query},
                message=f"Web lookup failed: {error}",
            )

        logger.info(
            "web.ask served query=%r backend=%s sources=%s",
            query,
            observed.get("backend"),
            observed.get("sources"),
        )
        return ToolResult(
            action_id=action.id,
            ok=True,
            code="ok",
            target="web",
            requested={"query": query},
            observed=observed,
            message="Web answer retrieved",
        )

    async def health(self) -> dict:
        return {
            "ok": True,
            "contractVersion": self.contract_version,
            "backend": self.backend,
            "braveConfigured": self._brave is not None,
            "geminiConfigured": self._gemini is not None,
        }

    async def close(self) -> None:
        if self._gemini is not None:
            await self._gemini.close()
        if self._brave is not None:
            await self._brave.close()
        await self._search.close()
