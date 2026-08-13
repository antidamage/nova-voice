"""Live-host fixtures for the voice suite.

This suite is not a unit test and deliberately does not pretend to be one. It
needs a running nova-voice with the test surface enabled, a resident STT/LLM/TTS
stack, and a reachable dashboard, so it is deselected from the ordinary `pytest`
run and skips loudly rather than failing when a host is not configured.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from .judge import Judge
from .runner import VoiceHarness

CASES_DIR = Path(__file__).parent / "cases"


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--voice-host",
        default=os.environ.get("NOVA_VOICE_TEST_HOST", ""),
        help="Base URL of a nova-voice host with test_harness_enabled (e.g. https://iridium:8766)",
    )
    parser.addoption(
        "--voice-cert",
        default=os.environ.get("NOVA_VOICE_TEST_CERT", ""),
        help="Client certificate PEM. The voice listener requires mTLS when a CA is configured.",
    )
    parser.addoption(
        "--voice-key",
        default=os.environ.get("NOVA_VOICE_TEST_KEY", ""),
        help="Client private key PEM for --voice-cert",
    )
    parser.addoption(
        "--dashboard-url",
        default=os.environ.get("NOVA_DASHBOARD_URL", "http://nova.local"),
        help="Base URL of the Nova dashboard, used as the truth reference for graded answers",
    )
    parser.addoption(
        "--llm-url",
        default=os.environ.get("NOVA_VOICE_LLM_URL", ""),
        help="Base URL of the llama-server used to grade answers",
    )
    parser.addoption(
        "--llm-model",
        default=os.environ.get("NOVA_VOICE_LLM_MODEL", "Qwen3.5-4B"),
        help="Model id for the grading pass",
    )


@pytest.fixture(scope="session")
def voice_host(request: pytest.FixtureRequest) -> str:
    host = str(request.config.getoption("--voice-host") or "").strip()
    if not host:
        pytest.skip("no --voice-host configured; the voice suite needs a live nova-voice")
    return host.rstrip("/")


@pytest.fixture(scope="session")
def voice_identity(request: pytest.FixtureRequest) -> tuple[str, str] | None:
    """The client certificate the voice listener requires when a CA is set."""

    cert = str(request.config.getoption("--voice-cert") or "").strip()
    key = str(request.config.getoption("--voice-key") or "").strip()
    return (cert, key) if cert and key else None


@pytest.fixture
async def harness(
    voice_host: str,
    judge_spec: dict[str, str],
    voice_identity: tuple[str, str] | None,
):
    connection = VoiceHarness.connect(
        voice_host,
        dashboard_url=judge_spec["dashboard_url"],
        identity=voice_identity,
    )
    # Every case starts from silence. A window left open by the previous case
    # would make its follow-ups addressed for free, which is exactly the
    # behaviour some of these cases exist to check.
    await connection.end_conversations()
    try:
        yield connection
    finally:
        await connection.end_conversations()
        await connection.close()


@pytest.fixture(scope="session")
def judge_spec(request: pytest.FixtureRequest) -> dict[str, str]:
    return {
        "llm_base_url": str(request.config.getoption("--llm-url") or "").strip(),
        "llm_model": str(request.config.getoption("--llm-model")),
        "dashboard_url": str(request.config.getoption("--dashboard-url")).rstrip("/"),
    }


@pytest.fixture
async def judge(judge_spec: dict[str, str]):
    if not judge_spec["llm_base_url"]:
        pytest.skip("no --llm-url configured; graded cases need the household language model")
    grader = Judge.connect(**judge_spec)
    try:
        yield grader
    finally:
        await grader.close()


def load_cases(name: str) -> list[dict]:
    return list(yaml.safe_load((CASES_DIR / name).read_text("utf-8")) or [])
