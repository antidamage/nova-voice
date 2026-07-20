# Rollback and stop procedure

Nova Voice is deliberately removable from the hot path. Stopping it does not
stop Home Assistant or the dashboard. (The legacy HA Assist/Wyoming voice path
has been retired; stopping Nova Voice leaves the household without voice
control until it is restarted.)

On Iridium, an administrator can stop only the new services:

```sh
sudo systemctl disable --now nova-voice.service nova-voice-llm.service
```

On a Linux satellite:

```sh
systemctl --user disable --now nova-voice-satellite.service
```

On Indium, unload only the Nova Voice LaunchAgent from the logged-in user
session; do not remove the signed app or microphone permission:

```sh
launchctl bootout "gui/$(id -u)/nz.co.skull.NovaVoiceSatellite"
```

Re-enabling the old household voice path is an operator decision and must be
verified on the dashboard/HA host before any satellite is migrated. This file
does not contain a command to disable that existing path.
