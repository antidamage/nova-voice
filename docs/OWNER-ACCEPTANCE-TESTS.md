# Owner acceptance tests

Run these as ordinary household use checks after a deployment. Record the date, outcome, and any unexpected wording or action in the
roadmap/evaluation notes. Browser microphone testing is in scope. Do not run a speaker-to-microphone loopback test unless you have
deliberately set it up; no such test is needed for this checklist.

## July 2026 voice-remediation acceptance pass

Run these from the browser microphone after loading the latest dashboard. Use a harmless made-up phone number and reversible device
actions. The diagnostic baseline intentionally keeps **Serena** while using **English**, **voice-native accent**, **100% speech rate**,
**0% pitch**, **natural emotion**, and **0% emotion mirroring**.

Record each item as Pass, Fail, or Not tested, along with the exact transcript/reply when it fails.

### Number and memory correctness

1. Ask, “What is the temperature?” when a result containing `9.8C` is available, or ask Nova to read “9.8C.” Expected: “nine point
   eight degrees,” never “nine point eight C.”
2. Ask Nova to read `2026`, `106th`, `105 Example Street`, and `0212345678`. Expected: “twenty twenty six,” “one hundred and sixth,”
   “one zero five,” and phone digits individually. The phone number must never contain “hundred” or “thousand.”
3. Say, “Save that my test phone number is 0212345678.” Then try each query separately: “Do you remember my phone number?”, “Check
   your memory,” and “What have you saved?” Expected: every query uses MemPalace and returns the saved digits accurately.
4. Ask about something never saved. Expected: “I don't have a saved memory about that,” rather than an invented memory.
5. Turn a light on, then check memory. Expected: the routine command and transient light state were not saved.

### Patient endpointing and resumed speech

1. Say, “Nova, remind me to call Sam and…” Pause for roughly two to three seconds, then add, “…ask about Friday.” Expected: Nova waits,
   shows one combined user statement, and handles the complete request once.
2. Make a clearly complete short request and stop. Expected: Nova answers promptly rather than imposing the full 3.5-second patient
   window on every turn.
3. Finish a statement, then quickly resume while Nova is still thinking: “Actually, add that it should be after lunch.” Expected: the
   first read-only attempt is cancelled, the addition is appended, and one combined request is reprocessed.
4. Perform a harmless reversible action, then add more information after the action has visibly occurred. Expected: Nova treats the
   addition as a contextual follow-up and does not repeat the first device action.
5. Add information while Nova has begun speaking. Expected: old playback stops cleanly; no stale sentence continues underneath the new
   turn.

### Browser listener recovery and duplicate-stream protection

1. With browser voice active, briefly disable and restore Wi-Fi. Do not reload. Expected: the browser satellite reconnects and accepts a
   new spoken turn after bounded backoff.
2. Background the browser or lock the iPad briefly, return to the dashboard, and speak again. Expected: suspended audio resumes or the
   microphone is reacquired without reloading.
3. If practical, revoke/re-enable the browser's microphone permission or switch the active microphone device. Expected: an ended track
   is reacquired; a denied permission remains visibly unavailable rather than silently pretending to listen.
4. Open the same dashboard/browser profile in two tabs so both advertise the same satellite identity. Expected: the newest connection
   supersedes the older one; a single utterance produces one transcript, one action, and one reply.
5. Trigger a reply and then interrupt it with another addressed request. Expected: only one audio stream is audible, the old stream is
   stopped, and the browser remains connected afterward.

### Voice consistency baseline

1. Ask Nova to repeat the same two-sentence passage three times. Expected: no conspicuous random plunge in pitch or speed within a reply
   or between repetitions.
2. Ask for a two- or three-sentence answer. Expected: sentence boundaries retain one coherent voice, pace, and mood rather than sounding
   like separate regenerated voices.
3. In Voice diagnostics/server health, inspect `ttsConsistency` after repeated passages. Expected: turns and tracked fixtures increase;
   stable repetitions do not increase the alert count. A failure report should include the approximate time and passage so its
   `tts_consistency` event can be found without retaining audio.
4. Leave the stable baseline in place during this pass. Reintroduce accent, rate, pitch, emotion, and mirroring one at a time only after
   the baseline passes, noting which single change reintroduces variation.

### Automation tests still pending

The earlier acceptance pass did not test automation. Complete all **Proactive home autonomy** tests below. Start with simulation and use
only a non-critical, reversible live condition. Automation remains Not tested until those checks are explicitly completed.

## Knowledge and spoken delivery

1. From a browser microphone, ask a current factual question without saying “search the web”, such as “Nova, what is the latest
   weather warning for Auckland?” Expected: if Nova lacks the answer, it performs one web lookup and gives a short sourced answer rather
   than ending with “I don’t know” or “I can’t do that.”
2. Ask Nova to speak: “The year is 2026; this is the 106th test.” Expected pronunciation: “twenty twenty-six” and “one hundred and
   sixth,” with natural surrounding speech.
3. Ask it to read a made-up address and phone number, for example “105 Example Street” and “021 234 5678.” Expected: digit-by-digit
   address/phone delivery (“one zero five”, not a cardinal number) while normal counting remains cardinal/ordinal.

## Selective conversational memory

1. In a normal longer chat, say one useful non-sensitive preference, e.g. “I prefer concise evening reminders because I’m tired then.”
   Continue the conversation, then ask “What do you remember about my reminder preference?” Expected: Nova recalls the salient
   preference concisely and accurately.
2. Issue a routine command such as “turn the lounge lights on,” then ask what Nova remembered. Expected: the light state/command was
   not saved as a conversational memory.
3. Offer a sensitive fact, such as a health, finance, security, or third-party detail. Expected: Nova asks for confirmation instead of
   using it immediately. In **Config → Voice agent authority → Selective conversational memory**, verify it appears under **Review
   required**, then confirm or discard it.
4. In the same dashboard section, test **Pin**, **Correct**, **Expiry**, and **Forget** on a harmless memory. Expected: each action is
   reflected after refresh; forgotten memories no longer appear or affect recall.
5. Select **Consolidate** and **Backup and verify**. Expected: both complete without silently rewriting a conflicting memory.

## Proactive home autonomy

1. In **Config → Voice agent authority → Household identities**, assign your recognized profile the **Owner** role. Expected: only an
   explicitly assigned owner can draft, approve, activate, or roll back an automation.
2. Under **Proactive home automations**, draft a harmless dashboard-channel rule (for example a `device_health` or `energy` event), then
   select **Simulate**. Expected: it reports a safe simulation and proposed-action count; no household device changes.
3. Select **Approve**, then **Activate**. Expected: state advances only in order: draft → simulated → approved → active. Select
   **Roll back** and confirm it becomes rolled back.
4. To exercise a live proposal, use a non-critical, reversible household condition that already produces a device-health, energy,
   occupancy, or household-state event. Expected: one reviewable intervention appears in **Proactive feedback**. Voice proposals route
   to the dashboard during quiet hours.
5. Mark a proposal **accepted**, **dismissed**, **redundant**, or **annoying**. Expected: feedback and final status persist after a
   refresh, with no automatic device mutation.
6. Ask Nova read-only household questions such as “What device needs attention?”, “What is using energy?”, “Who is home?”, or “What
   media is playing?” Expected: it uses the live dashboard state and does not invent a device, scene, timer, or schedule.

## Final health check

1. Refresh the dashboard and confirm Voice Agent administration and memory panels load without an unavailable banner.
2. Use the browser microphone for one ordinary smart-home command and one factual question. Expected: the normal voice path remains
   responsive; this is a user-run check only, not a speaker-to-microphone loopback test.
