"""Route one parsed client frame into the session manager.

Kept out of the socket handler so the transport and the message semantics can
be tested apart from each other — and so the in-memory tests drive exactly the
same dispatch the live endpoint does, rather than a reimplementation of it that
can quietly drift.

Deliberately synchronous: nothing here may block the receive loop. The one
message that does real work — a tool call — is handed off as a task by the
session manager rather than awaited inline.
"""

from __future__ import annotations

import logging

from nova_voice.companion.protocol import (
    ApprovalResponse,
    CompanionHello,
    Heartbeat,
    JobAccept,
    JobFailed,
    JobProgress,
    JobReject,
    JobResult,
    LogEvent,
    PersonalResult,
    TelemetryMessage,
    ToolCall,
)
from nova_voice.companion.session import CompanionSession, CompanionSessionManager

logger = logging.getLogger(__name__)

_LOG_LEVELS = {"debug": 10, "info": 20, "warning": 30, "error": 40}


def dispatch_companion_message(
    sessions: CompanionSessionManager,
    session: CompanionSession,
    message,
) -> None:
    if isinstance(message, TelemetryMessage):
        sessions.handle_telemetry(session, message.telemetry)
    elif isinstance(message, Heartbeat):
        sessions.handle_heartbeat(session)
    elif isinstance(message, JobAccept):
        sessions.handle_accept(session, message.job_id, message.attempt_id)
    elif isinstance(message, JobReject):
        sessions.handle_reject(
            session,
            message.job_id,
            message.attempt_id,
            message.reason,
            message.retry_after_seconds,
        )
    elif isinstance(message, JobProgress):
        sessions.handle_progress(session, message)
    elif isinstance(message, JobResult):
        sessions.handle_result(session, message.job_id, message.attempt_id, message.result)
    elif isinstance(message, JobFailed):
        sessions.handle_failed(
            session, message.job_id, message.attempt_id, message.reason, message.detail
        )
    elif isinstance(message, ToolCall):
        sessions.dispatch_tool_call(session, message)
    elif isinstance(message, PersonalResult):
        sessions.handle_personal_result(session, message)
    elif isinstance(message, ApprovalResponse):
        sessions.handle_approval_response(session, message.approval_id, message.approved)
    elif isinstance(message, LogEvent):
        # Structural by contract. Logged at the client's own level without ever
        # being read back as state.
        logger.log(
            _LOG_LEVELS[message.level],
            "companion event=%s detail=%s",
            message.event,
            message.detail,
        )
    elif isinstance(message, CompanionHello):
        # A second hello on an established session refreshes what the device
        # can currently do without tearing the session down.
        sessions.handle_telemetry(session, message.telemetry)
