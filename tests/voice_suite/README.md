# Voice suite

End-to-end spoken cases against a live nova-voice host. Deselected from the
ordinary `pytest` run — these need a resident model stack and take real seconds
each.

```bash
# On iridium, where the mTLS material and the resident models already live.
cd /opt/nova-voice/current
.venv/bin/python -m pytest -m "voice_suite and not slow" \
  --voice-host=https://127.0.0.1:8766 \
  --voice-cert=/etc/nova-voice/tls/client.crt \
  --voice-key=/etc/nova-voice/tls/client.key \
  --llm-url=http://127.0.0.1:8765/v1 \
  --dashboard-url=http://nova.local
```

The voice listener runs with `ssl_cert_reqs=2` whenever a CA is configured, so
`--voice-cert`/`--voice-key` are required — point them at any client identity
the voice CA signed (the dashboard's will do).

## How a case runs

1. The phrase is synthesized once via `POST /v1/voices/preview` and cached under
   `.clips/` (gitignored; `clip-manifest.json` is the readable index). Changing
   a phrase produces a new clip and leaves the old one alone.
2. The PCM goes to `POST /v1/test/turn` — the same entry a satellite's audio
   takes, so STT, wake matching, echo defence, dedup, speaker recognition,
   interpretation, policy and rendering all run for real.
3. The turn runs **dry**: every household request is built and reported, and
   none is sent. Cases assert on those withheld requests rather than on what
   Nova said, because the reply is a rendering choice and the request is the
   behaviour.

Nothing is played through a microphone or a speaker, and a full run changes
nothing in the house.

## Requirements on the host

- `NOVA_VOICE_TEST_HARNESS_ENABLED=true` in `/etc/nova-voice/nova-voice.env`.
  `deploy-nova-stack.ps1` seeds this once; turning it off keeps it off.
- Audio inference enabled (the suite needs STT).

## Adding a case

Single-phrase cases are data, in `cases/direct_commands.yaml`. Multi-turn cases
are code (`test_conversations.py`, `test_training_mode.py`) because their point
is sequencing and timing, which a table cannot express honestly.

- `expect` is deterministic and checked first — decision, tools, request paths,
  request bodies, resolved targets, colour, temperature direction.
- `judge` is only for what has no assertable form: whether a spoken answer is
  *true*. It runs against the household's own language model from an **empty**
  system prompt plus the case's own context, so it grades the rubric rather
  than re-deriving Nova's opinion. `context_from: clock` and
  `context_from: weather` build that context from a live reference rather than
  from the turn — a judge given only the turn can tell you an answer is
  coherent, never that it is correct.
- `mode: record` is for behaviour we have not decided on. The run captures what
  the stack did and skips, so the expectation gets written from evidence
  instead of from a guess.
