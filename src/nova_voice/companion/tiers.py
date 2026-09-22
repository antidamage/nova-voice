"""Map companion telemetry onto a participation tier.

The companion is optional by construction: everything it does, Iridium can
still do. The tier is how that stays true as the device's battery, thermal
state and process state change, and it exists mainly to stop the phone flapping
in and out of the routing table. Three guards do that:

* **Hysteresis.** Climbing back into a tier needs the battery to clear the
  threshold by a margin, so a reading hovering on a boundary cannot oscillate.
* **Dwell.** An *improvement* must hold for a sustained period before it is
  believed. A *degradation* applies immediately — a hot or nearly-flat phone
  should stop being offered work at once, not a minute later.
* **Staleness.** Telemetry that has stopped arriving is not evidence of health.
  Past the staleness window the tier is ``off`` regardless of the last reading.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from nova_voice.companion.protocol import CompanionTelemetry


class CompanionTier(StrEnum):
    FULL = "full"
    REDUCED = "reduced"
    ADVISORY = "advisory"
    OFF = "off"


# Worst first, so comparisons read as "at least this capable".
_TIER_ORDER: tuple[CompanionTier, ...] = (
    CompanionTier.OFF,
    CompanionTier.ADVISORY,
    CompanionTier.REDUCED,
    CompanionTier.FULL,
)


def tier_rank(tier: CompanionTier) -> int:
    return _TIER_ORDER.index(tier)


@dataclass(frozen=True)
class TierThresholds:
    off_below: float = 0.15
    advisory_below: float = 0.25
    reduced_below: float = 0.50
    hysteresis: float = 0.05
    dwell_seconds: float = 60.0
    # Telemetry older than this is not trusted to describe the device now.
    stale_after_seconds: float = 180.0


@dataclass(frozen=True)
class TierDecision:
    tier: CompanionTier
    reason: str
    changed: bool


def _raw_tier(
    telemetry: CompanionTelemetry, thresholds: TierThresholds
) -> tuple[CompanionTier, str]:
    """The tier the current reading argues for, before hysteresis or dwell."""

    if telemetry.thermal_state == "critical":
        return CompanionTier.OFF, "thermal state is critical"
    if not telemetry.models.hot_available:
        return CompanionTier.OFF, "no on-device model is available"
    if telemetry.battery < thresholds.off_below and not telemetry.charging:
        return CompanionTier.OFF, f"battery {telemetry.battery:.0%} is below the floor"
    if telemetry.thermal_state == "serious":
        return CompanionTier.ADVISORY, "thermal state is serious"
    if telemetry.low_power_mode:
        return CompanionTier.ADVISORY, "low power mode is on"
    if telemetry.charging:
        # Mains power removes the reason for every battery-derived limit.
        return CompanionTier.FULL, "charging"
    if telemetry.battery < thresholds.advisory_below:
        return CompanionTier.ADVISORY, f"battery {telemetry.battery:.0%} is low"
    if telemetry.battery < thresholds.reduced_below:
        return CompanionTier.REDUCED, f"battery {telemetry.battery:.0%} is below half"
    return CompanionTier.FULL, f"battery {telemetry.battery:.0%} on nominal thermals"


@dataclass
class TierTracker:
    """Stateful tier resolution across a stream of telemetry frames."""

    thresholds: TierThresholds = field(default_factory=TierThresholds)
    tier: CompanionTier = CompanionTier.OFF
    reason: str = "no telemetry received"
    last_update: float | None = None
    _pending_tier: CompanionTier | None = None
    _pending_reason: str = ""
    _pending_since: float | None = None

    def update(self, telemetry: CompanionTelemetry, *, now: float) -> TierDecision:
        first_reading = self.last_update is None
        self.last_update = now
        candidate, reason = _raw_tier(telemetry, self.thresholds)
        if first_reading:
            # Dwell exists to disbelieve a *change*, not to disbelieve the
            # first thing a freshly connected device says about itself.
            # Without this a healthy phone would be unusable for a whole dwell
            # window after every reconnect.
            self.tier = candidate
            self.reason = reason
            return TierDecision(tier=candidate, reason=reason, changed=True)
        if candidate == self.tier:
            self._clear_pending()
            return TierDecision(tier=self.tier, reason=self.reason, changed=False)

        if tier_rank(candidate) < tier_rank(self.tier):
            # Degradations are never delayed and never need headroom.
            self._clear_pending()
            self.tier = candidate
            self.reason = reason
            return TierDecision(tier=candidate, reason=reason, changed=True)

        if not self._clears_hysteresis(telemetry, candidate):
            self._clear_pending()
            return TierDecision(tier=self.tier, reason=self.reason, changed=False)

        if self._pending_tier != candidate:
            self._pending_tier = candidate
            self._pending_reason = reason
            self._pending_since = now

        assert self._pending_since is not None
        if now - self._pending_since < self.thresholds.dwell_seconds:
            return TierDecision(tier=self.tier, reason=self.reason, changed=False)

        self.tier = candidate
        self.reason = self._pending_reason
        self._clear_pending()
        return TierDecision(tier=self.tier, reason=self.reason, changed=True)

    def current(self, *, now: float) -> TierDecision:
        """The tier as of ``now``, accounting for telemetry that stopped."""

        if self.last_update is None:
            return TierDecision(CompanionTier.OFF, "no telemetry received", False)
        age = now - self.last_update
        if age > self.thresholds.stale_after_seconds:
            return TierDecision(
                CompanionTier.OFF, f"telemetry is {age:.0f}s stale", False
            )
        return TierDecision(self.tier, self.reason, False)

    def _clears_hysteresis(
        self, telemetry: CompanionTelemetry, candidate: CompanionTier
    ) -> bool:
        """Is the battery far enough past the boundary to climb into ``candidate``?"""

        if telemetry.charging:
            # Charging is a discrete fact, not a noisy reading near a boundary.
            return True
        floor = {
            CompanionTier.OFF: None,
            CompanionTier.ADVISORY: self.thresholds.off_below,
            CompanionTier.REDUCED: self.thresholds.advisory_below,
            CompanionTier.FULL: self.thresholds.reduced_below,
        }[candidate]
        if floor is None:
            return True
        return telemetry.battery >= floor + self.thresholds.hysteresis

    def _clear_pending(self) -> None:
        self._pending_tier = None
        self._pending_reason = ""
        self._pending_since = None
