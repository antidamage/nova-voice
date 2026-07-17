# Nova Voice

Nova Voice is the standalone, local voice runtime for the Nova household
dashboard. It runs on Iridium and talks to Nova only through its documented
HTTP/MCP interfaces; dashboard source is deliberately not imported or bundled.

Wake-word conversations retain room-local user/agent history until 20 seconds
of inactivity. Follow-up speech needs no repeated wake word. Verified dashboard
commands receive a short personality-aware confirmation; only the satellite that
heard the elected request plays it, while every dashboard continues to receive
the speaking animation.

Voice characteristics are owned by Nova's `GET /api/voice` contract. A dashboard
change calls `POST /v1/settings/refresh` on Iridium; the endpoint accepts no
settings payload, fetches the complete contract from Nova, and applies speaker,
language, accent, pace, pitch, baseline mood, and emotion mirroring live. The
same collection runs at service startup, while dashboard outages leave the
configured environment/persona defaults active.
Indium and Nocturnium are native, supervised audio satellites.

## Development quick start

```sh
cd nova-voice
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
cp .env.example .env
nova-voice preflight
pytest
```

The default development settings keep passive execution and model-backed audio
disabled. `nova-voice text "turn the lounge lights on" --wake` exercises the
text/control path once the local LLM and Nova endpoint are configured. Use the
deployment units under `deploy/` for supervised audio operation; the checked-in
defaults never silently enable capture or execution.

The checked-in runtime is intentionally development-safe: shadow mode is on,
transcripts are retained for at most 24 hours, and raw audio is never written
to disk. Physical microphones, model weights, TLS identities, and the Nova MCP
token must be provisioned before enabling a live deployment.

## Layout

- `src/nova_voice`: replaceable audio, inference, interpretation, session,
  capability-provider, satellite, and retention adapters
- `skills/`: compact instructions supplied to the deployed local LLM
- `config/`: persona and satellite environment examples (no secrets)
- `docs/`: contracts, architecture, model/VRAM evidence, rollout, and tests
- `ops/`: pinned model preparation, preflight, and smoke-test scripts

The opt-in development microphone and response inspector is documented in
[`docs/DIAGNOSTICS.md`](docs/DIAGNOSTICS.md). It is served by Nova Voice over
the existing authenticated endpoint and is never part of the dashboard.

All inference remains local to the household LAN. Raw audio is held only in
bounded memory; development transcripts expire after 24 hours.

## Private deployment reference

Concrete household details (hostnames, LAN addresses, account names, signing
identities) are deliberately absent from this repository. Documentation refers
to them as `PRIVATEREF.md#<section>`; that file is git-ignored and lives only
on household machines. Copy your own values into a local `PRIVATEREF.md` when
deploying.
