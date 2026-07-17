from __future__ import annotations

import struct
from enum import IntEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

MAGIC = b"NVAF"
PROTOCOL_VERSION = 1
HEADER = struct.Struct("!4sBBHQQI")
FLAG_PLAYBACK_ACTIVE = 1 << 0


class FrameKind(IntEnum):
    AUDIO_INPUT = 1
    AUDIO_OUTPUT = 2
    CONTROL = 3


class SatelliteCapabilities(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    microphone: bool = True
    speaker: bool = True
    echo_cancellation: bool = Field(default=False, alias="echoCancellation")
    noise_suppression: bool = Field(default=False, alias="noiseSuppression")
    automatic_gain_control: bool = Field(default=False, alias="automaticGainControl")
    # Newer clients acknowledge the moments output actually starts and finishes.
    # The default keeps protocol-v1 clients that predate those events compatible.
    playback_events: bool = Field(default=False, alias="playbackEvents")


class SatelliteHello(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    protocol_version: int = Field(alias="protocolVersion")
    satellite_id: str = Field(alias="satelliteId", min_length=1)
    display_name: str = Field(alias="displayName", min_length=1)
    room_id: str = Field(alias="roomId", min_length=1)
    client: Literal["linux-native", "macos-native"]
    supervisor: Literal["systemd", "launchd"]
    capture_policy: str = Field(alias="capturePolicy")
    dashboard_foreground: bool | None = Field(default=None, alias="dashboardForeground")
    capabilities: SatelliteCapabilities

    def validate_protocol(self) -> None:
        if self.protocol_version != PROTOCOL_VERSION:
            raise ValueError(f"unsupported protocol version: {self.protocol_version}")
        if self.capture_policy != "always":
            raise ValueError("native v1 satellites must use always capture")
        if not self.capabilities.microphone:
            raise ValueError("a v1 satellite must advertise a microphone")
        if not self.capabilities.speaker:
            raise ValueError("a v1 satellite must advertise a speaker")
        if self.client == "macos-native" and self.supervisor != "launchd":
            raise ValueError("macOS satellites must be supervised by launchd")
        if self.client == "linux-native" and self.supervisor != "systemd":
            raise ValueError("Linux satellites must be supervised by systemd")


class AudioFrame(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    kind: FrameKind
    flags: int = 0
    sequence: int
    monotonic_ns: int
    payload: bytes

    def pack(self) -> bytes:
        header = HEADER.pack(
            MAGIC,
            PROTOCOL_VERSION,
            int(self.kind),
            self.flags,
            self.sequence,
            self.monotonic_ns,
            len(self.payload),
        )
        return header + self.payload

    @classmethod
    def unpack(cls, value: bytes) -> AudioFrame:
        if len(value) < HEADER.size:
            raise ValueError("satellite frame is shorter than its header")
        magic, version, kind, flags, sequence, monotonic_ns, payload_length = HEADER.unpack_from(
            value
        )
        if magic != MAGIC or version != PROTOCOL_VERSION:
            raise ValueError("invalid satellite frame magic/version")
        payload = value[HEADER.size :]
        if len(payload) != payload_length:
            raise ValueError("satellite frame payload length mismatch")
        return cls(
            kind=FrameKind(kind),
            flags=flags,
            sequence=sequence,
            monotonic_ns=monotonic_ns,
            payload=payload,
        )
