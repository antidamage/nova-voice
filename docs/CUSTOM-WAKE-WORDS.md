# Custom wake words with the deployed NeMo STT

Assessment written 2026-07-18 against the pinned production stack:
`nvidia/nemotron-speech-streaming-en-0.6b` (cache-aware FastConformer with a
pure RNNT decoder) on `nemo_toolkit[asr]==2.7.3`, decoded greedily through
`NemoSpeechToText` (`src/nova_voice/inference/stt.py`).

## The problem

Wake matching is transcript-first (`WakePhraseMatcher`), so a wake word only
works if the ASR actually emits it. The RNNT decoder's internal language model
does not misspell invented names — it deletes them. "Okay Beemo what time is
it" transcribes as "Okay, what time is it". This affects the word anywhere in
the utterance, so it blocks both wake detection and the LLM ever seeing the
name conversationally. Real English words ("bandit") work; invented names
("beemo") are unreliable-to-invisible.

The previously wired openWakeWord acoustic detector has been removed. It could
only fire a wake trigger; it never fixed the transcript, so it could not make
a name usable conversationally, and it required training a new acoustic model
per wake word — the opposite of "user types a new wake word in the dashboard".

## What the model supports without retraining

NeMo ships three decode-time context-biasing mechanisms
([docs](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/asr_customization/word_boosting.html)):

| Method | Model types | Usable here? |
| --- | --- | --- |
| GPU-accelerated phrase boosting (GPU-PB / [TurboBias](https://arxiv.org/abs/2508.07014)) | CTC, RNNT/TDT, AED; greedy and beam | **Yes — the only applicable option** |
| Flashlight lexicon boosting | CTC lexicon decoding only | No (pure RNNT model) |
| CTC-based word spotter (CTC-WS) | needs a CTC head/hybrid model | No (pure RNNT model) |

GPU-PB is the one that fits. It builds a phrase-boosting tree from a list of
key phrases, tokenized with the model's own BPE vocabulary, and applies a
shallow-fusion score bonus during token selection in greedy decoding. Because
it biases subword paths, it can surface names the model never saw in training
— that is exactly the OOV-deletion counter we need. It is configured on the
decoding config (`rnnt_decoding.strategy="greedy_batch"` plus a
`boosting_tree` section: `key_phrases_list`/`key_phrases_file`,
`context_score` ≈ 1.0, `depth_scaling` ≈ 2.0, and a tunable
`boosting_tree_alpha`), applied via `change_decoding_strategy` — no weights
change, so it composes with the dashboard `wakeWords` list at settings-refresh
time.

## Integration sketch

1. On startup and on `/v1/settings/refresh`, take the dashboard `wakeWords`
   list (plus the agent display name), build the boosting-tree config, and
   call `change_decoding_strategy` on the loaded model inside the GPU
   execution gate.
2. Nothing else changes: `WakePhraseMatcher` already accepts the boosted
   spellings, and boosted transcripts flow to the LLM unchanged, which covers
   conversational recognition too.
3. Keep the alias list (`bimo`, `bemo`, `beamo`…) — boosting shifts
   probability, it does not guarantee one canonical spelling.

## Reliability expectations

- Published results: +8–10 % absolute key-phrase F-score under greedy
  decoding, with roughly 2–5 % RTFx overhead and negligible WER cost at small
  list sizes (ours is < 20 phrases; degradation was only measured at ~20 000).
- False positives are the tuning axis: over-boosted "beemo" will start
  claiming acoustically nearby speech ("be more", "bee no"). `context_score` /
  alpha must be tuned on recorded household audio, and the existing guards
  (article guard, early-position rule, dedup, echo guard) already bound the
  blast radius of a spurious wake token.
- **The open risk is streaming.** Neither the NeMo docs nor the TurboBias
  paper state how the boosting-tree state interacts with our cache-aware
  chunked decode (`partial_hypotheses` carried across 160 ms chunks). If tree
  state resets at chunk boundaries, a wake word split across a boundary may
  lose its boost. This is a prototype-and-measure question, not a blocker:
  worst case, boosting still applies cleanly to the final-utterance decode
  path, which is what the wake matcher consumes.
- Honest bottom line: for a short invented name, expect "usably reliable after
  tuning", not acoustic-wake-model reliability. A name whose subword pieces
  are common ("bee"+"mo") should recover well; a name hostile to the BPE
  inventory may stay marginal at any safe boost level.

## Deployment effort

- Prototype (offline, iridium, recorded clips of the target name at
  near/far/noisy conditions, sweep alpha): about half a day including the
  streaming-boundary measurement.
- Production wiring (settings-refresh hook, health surfacing of the active
  boost list, e2e via `ops/e2e/voice_e2e.py`): 1–2 days including deploy and
  live verification.
- No new dependencies, no model download, no service topology change; the
  boost list rebuild on settings refresh is fast (tree build is trivial at
  this list size).

## Retraining (out of scope — outline only)

If decode-time boosting proves insufficient for a wanted name, the escalation
is fine-tuning the ASR itself:

1. Build a corpus of the name: a few hundred utterances minimum, mixing real
   recordings from household mics with TTS-synthesized variants (multiple
   voices/rates), mixed into command templates and noise/far-field
   augmentation, plus a large replay set of general English to prevent
   forgetting.
2. Fine-tune the RNNT (full or adapter-based) in NeMo on the pinned
   checkpoint; validate WER on general speech does not regress and the
   cache-aware streaming export still works.
3. Re-run the residency gate (FP16 conversion, VRAM headroom) and the full
   wake/negative-contrast benchmark corpus from `docs/MODELS.md`.

Realistic cost: days of engineering plus GPU training time, repeated for every
new name — which is why it is out of scope for "user sets a wake word in the
dashboard". Decode-time boosting is the supported path; retraining is the
last resort for a name the tokenizer genuinely cannot be steered toward.
