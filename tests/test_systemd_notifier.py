from __future__ import annotations

import os

from nova_voice.satellites.systemd import SystemdNotifier


def test_systemd_notifier_is_a_noop_outside_a_service() -> None:
    notifier = SystemdNotifier({})

    assert notifier.watchdog_interval is None
    assert notifier.notify("READY=1") is False


def test_systemd_watchdog_uses_half_the_configured_interval() -> None:
    notifier = SystemdNotifier(
        {
            "NOTIFY_SOCKET": "@nova-test",
            "WATCHDOG_PID": str(os.getpid()),
            "WATCHDOG_USEC": "30000000",
        }
    )

    assert notifier.watchdog_interval == 15


def test_systemd_watchdog_ignores_another_process() -> None:
    notifier = SystemdNotifier(
        {
            "NOTIFY_SOCKET": "@nova-test",
            "WATCHDOG_PID": str(os.getpid() + 1),
            "WATCHDOG_USEC": "30000000",
        }
    )

    assert notifier.watchdog_interval is None
