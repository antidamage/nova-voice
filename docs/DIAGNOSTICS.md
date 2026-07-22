# Voice diagnostics page

Nova Voice includes an opt-in development page at
`https://<voice-server-host>:8766/diagnostics` (hostname: see
`PRIVATEREF.md#1.1`). It is owned and served by Nova Voice;
it does not import, modify, or depend on the Nova dashboard. It is a
push-to-record test client, distinct from both the always-on native Indium and
Nocturnium satellites and the supported dashboard browser satellite.

The page exercises the deployed resident path:

```text
browser microphone -> 16 kHz PCM16 -> STT -> interpretation/policy/persona
                   -> TTS -> browser playback
```

It displays the transcript, transcript confidence, probable emotion, speech
act, addressing probability, decision, shadow-policy outcome, response text,
tone instruction, and component timings. Audio is bounded in request memory and
is never persisted. The resulting development transcript uses the same 24-hour
retention policy as satellite turns.

## Enable it safely

The page is disabled by default and refuses to process a turn unless Nova Voice
is in shadow mode. The existing server requires a household-CA client
certificate before it serves either the page or its API.

1. On Iridium, create a temporary password file and issue the dedicated browser
   identity without replacing the existing CA:

   ```sh
   sudo /opt/nova-voice/current/ops/issue-satellite-identity.sh \
     browser-diagnostics /secure-transfer/browser-diagnostics \
     /root/browser-diagnostics-p12-password
   ```

2. Transfer `browser-diagnostics.p12` and `ca.crt` through an
   administrator-controlled path. Import the P12 into the current user's client
   certificate store. Trust `ca.crt` for the current user only if the browser
   does not already trust Iridium's server certificate. This intentionally
   trusts certificates issued by the household Nova Voice CA, so remove that
   trust after testing if the machine is not a permanent administrator client.
3. Set these values in `/etc/nova-voice/nova-voice.env` on Iridium:

   ```sh
   NOVA_VOICE_DIAGNOSTICS_ENABLED=true
   NOVA_VOICE_DIAGNOSTICS_MAX_AUDIO_SECONDS=30
   ```

   Keep `NOVA_VOICE_SHADOW_MODE=true`, then restart only
   `nova-voice.service`.
4. Open `https://<voice-server-host>:8766/diagnostics`, select the client identity if
   prompted, choose a microphone, and press the central button. Press it again
   after speaking; capture also stops automatically at the configured limit.

Browser microphone capture and AudioWorklet require a trusted secure context.
The page therefore does not provide an insecure HTTP fallback.

## Useful test phrases

Keep **Affirmative test turn** enabled for normal conversational tests. Disable
it only when checking passive-addressing thresholds.

| Phrase | Expected inspection |
| --- | --- |
| “Nova, turn the lounge light on.” | directive, addressed, shadowed action |
| “I want the air con to be on.” | desired state, shadowed action |
| “I gotta turn the air con on.” | self-intention, no action and usually no spoken response |
| “Why is the air con off?” | question/reply, not a control action |
| “Turn it off.” | clarification unless the active room goal has one salient target |
| Calmly: “Could you lower the lights?” | calm tone and calm response instruction |
| Angrily: “Turn that bloody heater off.” | angry/grumpy estimate with bounded matching delivery |
| Excitedly: “That worked! What else can you do?” | excited/social reply and perceptibly energetic TTS |

Use the structured details disclosure to capture evidence without copying
audio. A no-response result is valid for ignored ambient/self-intention speech.

## Disable and remove access

Set `NOVA_VOICE_DIAGNOSTICS_ENABLED=false` and restart only
`nova-voice.service`. Remove the `browser-diagnostics` client identity and any
user-level trust added for the household CA from the test machine. Disabling the
page does not stop either native satellite or modify the dashboard/HA stack.

## Browser standards

The page uses `getUserMedia` constraints for echo cancellation, noise
suppression, and automatic gain control as defined by the W3C Media Capture and
Streams specification. Capture is converted to PCM in an `AudioWorklet`, which
MDN documents as a secure-context API:

- https://www.w3.org/TR/mediacapture-streams/
- https://developer.mozilla.org/en-US/docs/Web/API/BaseAudioContext/audioWorklet
