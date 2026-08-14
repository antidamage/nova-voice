# Boosting household vocabulary in STT — tried, measured, reverted

**Result: it does not work. Do not re-try it without new evidence.**

The recognizer mishears the household's own nouns — "is the heater on" comes
back as "is the hero", "in the bedroom" as "in the paper". The stack already
has a decode-time boosting tree (`stt_context_biasing_enabled`,
`stt_boost_alpha`), used for the wake words and the agent name precisely
because an invented name would otherwise be deleted. Extending it to the device
manifest looks like the obvious fix.

It was implemented (`NovaProvider.spoken_vocabulary()` deriving rooms, zone
groups, climate controls and device names from the live manifest) and measured
against the same seven questions, dry, on the live host.

## Measurements

**Read these as indicative, not conclusive.** Two consecutive runs of the same
configuration were byte-identical, which I initially took as proof the results
were deterministic. A later run of the *reverted* configuration then produced
"Is the heater on?" — correct — where the first run of that same configuration
had produced "Is the hero?". So transcription varies between runs even with the
configuration fixed, and single runs cannot support fine distinctions between
these columns.

That variance is itself the most useful thing here: **you cannot A/B this
recognizer with one run per arm.** Anything revisiting this needs repeated runs
per configuration and a word-error rate, not a side-by-side of one sample each.

| asked | no biasing | 96 terms | 8 terms (rooms first) |
|---|---|---|---|
| "What is the temperature in here" | ✅ in here | ✅ in here | ❌ dropped |
| "What is the temperature outside" | ❌ dropped | ❌ dropped | ❌ dropped |
| "Tell me the weather" | ✅ | ❌ "Tell me the" | ✅ |
| "Tell me the time" | ✅ | ❌ "Told me the" | ✅ |
| "Are the lounge lights on" | "Are the lounge lights?" | "With a lounge lights." | "At a lounge lights." |
| "Is the heater on" | "Is the hero?" | ✅ **"Is the heater"** | ❌ "Is that you are uh" |
| "How warm is it in the bedroom" | "…in the paper?" | ❌ "A warm is in." | "…in a paper?" |

## Why it was reverted

The tree is a **decoding bias, not a dictionary**, and every added term pulls
ordinary speech toward it. Across the runs above, no configuration was shown to
beat plain wake-word biasing — each appeared to trade one word for another.

Given the run-to-run variance, the honest statement is not "biasing makes it
worse" but "**biasing was not shown to make it better**". Shipping a change to
a live household system on that basis is not justified, so it was reverted.
A properly powered experiment could still find a win.

Anything that revisits this needs a different mechanism, not a different list —
a per-phrase weight, a much lower alpha for household terms than for the agent
name, or biasing applied only once a turn is known to be a household command.

## Separate finding, worth keeping

Part of the apparent STT failure was **measurement error in the audit harness,
not the stack**. `ops/companion_answer_audit.py` and `ops/companion_baseline.py`
resampled the TTS engine's 32 kHz output to 16 kHz by dropping every Nth
sample, aliasing everything above the new Nyquist into the speech band —
exactly what `tests/voice_suite/clips.py` warns about in its own resampler.
Fixing them to the same box-average filter recovered "in here" and turned "But
it lounge loads" into "Are the lounge lights", with no change to Nova at all.

Any future harness that feeds synthesized audio to STT must filter on
downsample, or it measures itself.

## Still open

- Trailing words are dropped regardless of biasing ("on", "outside"). The
  suite pads 600 ms of trailing silence for this reason; it may need more.
- "bedroom" → "paper" survives every configuration tried here.
