"""Filesystem locations for the training subsystem.

Environment-overridable so tests and non-production hosts do not have to write
under /opt, but the defaults match what ops/install-trained-tts.sh provisions.
"""

from __future__ import annotations

import os

TRAINING_ROOT = os.environ.get("NOVA_TRAINING_ROOT", "/opt/nova-voice/training-sets")
"""One directory per training set."""

GPTSOVITS_ROOT = os.environ.get("GPTSOVITS_ROOT", "/opt/nova-voice/gptsovits/GPT-SoVITS")

GPTSOVITS_PYTHON = os.environ.get(
    "GPTSOVITS_PYTHON",
    "/opt/nova-voice/gptsovits/conda/bin/python",
)
"""Interpreter inside GPT-SoVITS's own conda env.

Deliberately under /opt, NOT in a user's home. nova-voice.service runs with
``ProtectHome=yes``, which replaces /home with an empty tmpfs inside the
service's mount namespace -- an interpreter there is invisible to the service
and to the training worker it spawns, failing with a confusing
``PermissionError`` that no chmod or group membership can fix. A system service's
dependencies belong in a system location.
"""

TRAINED_VOICES_DIR = os.environ.get("TRAINED_VOICES_DIR", "/opt/nova-voice/trained-voices")
"""Where a packaged bundle is published to become a selectable voice."""

TRAINING_MODE_STATE = os.environ.get(
    "NOVA_TRAINING_MODE_STATE", "/var/lib/nova-voice/training-mode.json"
)
"""Presence marks training mode active and records what to restore afterwards."""
