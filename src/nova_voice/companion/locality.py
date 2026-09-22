"""Classify where a companion actually connected from.

This is a security gate, not a convenience. Transparent replacement of
Iridium's reasoning — and the microphone role entirely — is permitted only on
the home LAN, so "am I at home?" must be answered from something the phone
cannot assert. The peer address of the socket is that something.

A phone on home Wi-Fi reaches the configured LAN endpoint and lands in a
configured home subnet. The same phone on cellular reaches the tailnet
endpoint and lands in the tailnet range, *even though its own radio state might
suggest otherwise*. The device's reported network name is diagnostic only: a
compromised or buggy client claiming ``atHome`` cannot unlock a home-LAN route.
"""

from __future__ import annotations

import ipaddress
import logging
from dataclasses import dataclass

from nova_voice.companion.protocol import Locality

logger = logging.getLogger(__name__)

# Tailscale's CGNAT range. Traffic arriving from here reached us over the
# tailnet regardless of what carried it underneath.
DEFAULT_TAILNET_SUBNETS = ("100.64.0.0/10", "fd7a:115c:a1e0::/48")


def _networks(values) -> tuple:
    networks = []
    for value in values:
        text = str(value).strip()
        if not text:
            continue
        try:
            networks.append(ipaddress.ip_network(text, strict=False))
        except ValueError:
            logger.warning("ignoring malformed companion subnet: %s", text)
    return tuple(networks)


@dataclass(frozen=True)
class LocalityClassifier:
    home_subnets: tuple = ()
    tailnet_subnets: tuple = ()

    @classmethod
    def from_settings(cls, home_subnets, tailnet_subnets=None) -> LocalityClassifier:
        return cls(
            home_subnets=_networks(home_subnets or ()),
            tailnet_subnets=_networks(tailnet_subnets or DEFAULT_TAILNET_SUBNETS),
        )

    @property
    def configured(self) -> bool:
        return bool(self.home_subnets)

    def classify(self, peer_host: str | None) -> Locality:
        if not peer_host:
            return "other"
        try:
            address = ipaddress.ip_address(peer_host.strip().strip("[]"))
        except ValueError:
            return "other"
        # Loopback is the reference client and the dashboard proxy on Iridium
        # itself, which is as "at home" as anything can be.
        if address.is_loopback:
            return "home_lan"
        if any(address in network for network in self.tailnet_subnets):
            return "tailnet"
        if any(address in network for network in self.home_subnets):
            return "home_lan"
        return "other"
