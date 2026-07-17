from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import httpx

from nova_voice.bootstrap import build_service
from nova_voice.config import get_settings
from nova_voice.domain import Utterance
from nova_voice.logging import configure_logging


async def _run_text(args: argparse.Namespace) -> int:
    settings = get_settings()
    service = build_service(settings)
    try:
        await service.initialize()
        utterance = Utterance.text(
            args.text,
            room_id=args.room,
            wake_detected=args.wake,
        )
        result = await service.handle(utterance)
        print(result.model_dump_json(indent=2))
        return 0
    finally:
        await service.close()


async def _preflight() -> int:
    settings = get_settings()
    checks: dict[str, dict] = {}
    async with httpx.AsyncClient(timeout=5) as client:
        for name, url in {
            "nova": f"{settings.nova_base_url.rstrip('/')}/api/version",
            "llm": f"{settings.llm_base_url.rstrip('/')}/models",
        }.items():
            try:
                response = await client.get(url)
                checks[name] = {"ok": response.is_success, "status": response.status_code}
            except httpx.HTTPError:
                checks[name] = {"ok": False}
    print(json.dumps(checks, indent=2))
    return 0 if all(check["ok"] for check in checks.values()) else 1


async def _run_satellite() -> int:
    from nova_voice.satellites.client import NativeSatelliteClient, SatelliteSettings

    await NativeSatelliteClient(SatelliteSettings()).run_forever()
    return 0


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(prog="nova-voice")
    subparsers = parser.add_subparsers(dest="command", required=True)

    text_parser = subparsers.add_parser("text", help="Run one text utterance through the service")
    text_parser.add_argument("text")
    text_parser.add_argument("--room", default="lounge")
    text_parser.add_argument("--wake", action="store_true")

    subparsers.add_parser("preflight", help="Check Nova and the local LLM endpoint")
    subparsers.add_parser("serve", help="Run the Nova Voice API")
    subparsers.add_parser("satellite", help="Run a supervised native audio satellite")
    health_parser = subparsers.add_parser(
        "satellite-health", help="Read the satellite's local redacted health file"
    )
    health_parser.add_argument("--path", type=Path)
    args = parser.parse_args()

    if args.command == "text":
        raise SystemExit(asyncio.run(_run_text(args)))
    if args.command == "preflight":
        raise SystemExit(asyncio.run(_preflight()))
    if args.command == "serve":
        import os

        import uvicorn

        settings = get_settings()
        uvicorn.run(
            "nova_voice.api:app",
            # Diagnostic escape hatch: uvicorn transport-level logging without
            # a code change (e.g. NOVA_VOICE_UVICORN_LOG_LEVEL=trace) for
            # chasing WS handshake wedges that leave no INFO-level trail.
            log_level=os.environ.get("NOVA_VOICE_UVICORN_LOG_LEVEL") or None,
            host=settings.host,
            port=settings.port,
            # Every uvicorn WS implementation "wedged" the same way after an
            # abrupt TLS disconnect mid-stream: legacy websockets 400-rejected
            # later upgrades with NUL-corrupted request lines, sansio silently
            # never answered them, wsproto 400s them.  The shared component is
            # the httptools HTTP parser that fronts the upgrade — pin the
            # pure-Python h11 parser instead.  ws stays on wsproto, which
            # answers handshakes immediately rather than hanging when
            # something upstream is wrong.
            http="h11",
            ws="wsproto",
            # Satellites continuously transmit PCM. The default 20-second
            # WebSocket ping makes macOS URLSession tear down an otherwise
            # active stream, so liveness comes from the audio transport.
            ws_ping_interval=None,
            ws_ping_timeout=None,
            ssl_certfile=str(settings.tls_cert_path) if settings.tls_cert_path else None,
            ssl_keyfile=str(settings.tls_key_path) if settings.tls_key_path else None,
            ssl_ca_certs=str(settings.tls_ca_path) if settings.tls_ca_path else None,
            ssl_cert_reqs=2 if settings.tls_ca_path else 0,
        )
    if args.command == "satellite":
        raise SystemExit(asyncio.run(_run_satellite()))
    if args.command == "satellite-health":
        from nova_voice.satellites.client import SatelliteSettings, read_health

        path = args.path or SatelliteSettings().health_path
        print(json.dumps(read_health(path), indent=2))


if __name__ == "__main__":
    main()
