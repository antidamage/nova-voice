from __future__ import annotations

import asyncio
import os
import socket
from collections.abc import Mapping


class SystemdNotifier:
    """Minimal sd_notify client for the supervised Linux satellite.

    No systemd package is needed: notify messages are small AF_UNIX datagrams.
    On macOS or an interactive shell, NOTIFY_SOCKET is absent and every method
    becomes a harmless no-op.
    """

    def __init__(self, environment: Mapping[str, str] | None = None) -> None:
        selected = environment if environment is not None else os.environ
        self._notify_socket = selected.get("NOTIFY_SOCKET")
        watchdog_pid = selected.get("WATCHDOG_PID")
        watchdog_usec = selected.get("WATCHDOG_USEC")
        self.watchdog_interval: float | None = None
        try:
            pid_matches = not watchdog_pid or int(watchdog_pid) == os.getpid()
        except ValueError:
            pid_matches = False
        if watchdog_usec and pid_matches:
            try:
                configured = int(watchdog_usec) / 1_000_000
            except ValueError:
                configured = 0
            if configured > 0:
                self.watchdog_interval = max(0.5, configured / 2)

    def notify(self, message: str) -> bool:
        if not self._notify_socket:
            return False
        address = self._notify_socket
        if address.startswith("@"):
            address = "\0" + address[1:]
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as client:
                client.sendto(message.encode("utf-8"), address)
            return True
        except OSError:
            return False

    def ready(self, status: str) -> bool:
        return self.notify(f"READY=1\nSTATUS={status}")

    async def run_watchdog(self) -> None:
        if self.watchdog_interval is None:
            return
        while True:
            await asyncio.sleep(self.watchdog_interval)
            self.notify("WATCHDOG=1")
