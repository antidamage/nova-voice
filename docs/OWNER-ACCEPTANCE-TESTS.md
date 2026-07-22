# Owner acceptance tests

Run these as ordinary household use checks after a deployment. Record the date, outcome, and any unexpected wording or action in the
roadmap/evaluation notes. Browser microphone testing is in scope. Do not run a speaker-to-microphone loopback test unless you have
deliberately set it up; no such test is needed for this checklist.

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
