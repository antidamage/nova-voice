"""Approving a companion-proposed mutation, and executing it exactly once.

NPT-406. The cases that matter are the awkward ones, so they are the tests:
a duplicate tap, an answer that arrives after the deadline, a forged
signature, a captured "yes" replayed against a different proposal, and a
server restart in the gap between the decision and the write.
"""

from __future__ import annotations

import base64
from datetime import UTC, datetime, timedelta

import pytest
from conftest import issue_certificate
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.hashes import SHA256

from nova_voice.companion.approvals import ApprovalError, CompanionApprovalGate
from nova_voice.companion.auth import (
    AuthenticatedIdentity,
    AuthenticationError,
    approval_material,
)
from nova_voice.durable.models import CompanionApprovalRecord
from nova_voice.durable.models import CompanionApprovalState as State
from nova_voice.durable.store import DurableAgentStore


@pytest.fixture
def device():
    """A household identity that can sign, and the identity Nova sees."""

    key, certificate = issue_certificate("companion-1")
    identity = AuthenticatedIdentity(
        identity="companion-1",
        roles=("companion",),
        not_after=certificate.not_valid_after_utc,
        fingerprint="test",
        certificate=certificate,
    )

    def sign(approval_id: str, approved: bool, nonce: str) -> str:
        material = approval_material(
            approval_id=approval_id, approved=approved, nonce=nonce
        )
        return base64.b64encode(
            key.sign(material, ec.ECDSA(SHA256()))
        ).decode()

    return identity, sign


class _Executor:
    """Counts how many times the mutation was actually performed."""

    def __init__(self, fail: bool = False) -> None:
        self.calls: list[str] = []
        self.fail = fail

    async def __call__(self, record: CompanionApprovalRecord) -> str:
        self.calls.append(record.idempotency_key)
        if self.fail:
            raise RuntimeError("the heater did not answer")
        return f"result-{len(self.calls)}"


async def _gate(tmp_path, executor: _Executor, name="durable.sqlite3"):
    store = DurableAgentStore(tmp_path / name)
    await store.initialize()
    return CompanionApprovalGate(store, execute=executor), store


async def _proposed(gate) -> CompanionApprovalRecord:
    return await gate.propose(
        provider="nova",
        tool="climate_set",
        arguments={"entity": "climate.panel_heater_2", "temperature": 18},
        summary="Set the panel heater to 18 degrees",
    )


# -- the record ---------------------------------------------------------------


async def test_a_proposal_stores_the_exact_target_not_a_description(tmp_path):
    # Approving a sentence must execute the action described at the moment it
    # was described, not whatever the sentence would resolve to later.
    gate, _ = await _gate(tmp_path, _Executor())
    record = await _proposed(gate)

    assert record.provider == "nova"
    assert record.tool == "climate_set"
    assert record.arguments == {"entity": "climate.panel_heater_2", "temperature": 18}
    assert record.status is State.PENDING


async def test_the_retention_horizon_outlives_the_answering_deadline(tmp_path):
    """Otherwise an unanswered proposal is deleted instead of recorded."""

    gate, _ = await _gate(tmp_path, _Executor())
    record = await _proposed(gate)

    assert record.expires_at is not None
    assert record.expires_at > record.respond_by


async def test_a_proposal_must_outlive_its_own_deadline():
    now = datetime.now(UTC)
    with pytest.raises(ValueError, match="outlive its answering deadline"):
        CompanionApprovalRecord(
            id="a1",
            provider="nova",
            tool="climate_set",
            summary="anything",
            idempotency_key="key-1",
            nonce="nonce-nonce-nonce",
            respond_by=now + timedelta(minutes=10),
            expires_at=now + timedelta(minutes=1),
        )


# -- deciding -----------------------------------------------------------------


async def test_an_approved_mutation_runs_once(tmp_path, device):
    identity, sign = device
    executor = _Executor()
    gate, _ = await _gate(tmp_path, executor)
    record = await _proposed(gate)

    decided = await gate.decide(
        record.id,
        approved=True,
        identity=identity,
        signature=sign(record.id, True, record.nonce),
    )
    assert decided.status is State.APPROVED

    done = await gate.execute_approved(record.id)

    assert done.status is State.EXECUTED
    assert done.result_ref == "result-1"
    assert executor.calls == [record.idempotency_key]


async def test_a_denied_mutation_never_runs(tmp_path, device):
    identity, sign = device
    executor = _Executor()
    gate, _ = await _gate(tmp_path, executor)
    record = await _proposed(gate)

    decided = await gate.decide(
        record.id,
        approved=False,
        identity=identity,
        signature=sign(record.id, False, record.nonce),
    )

    assert decided.status is State.DENIED
    with pytest.raises(ApprovalError, match="not approved"):
        await gate.execute_approved(record.id)
    assert executor.calls == []


async def test_a_duplicate_tap_does_not_change_the_answer(tmp_path, device):
    # A second "no" after a "yes" would make the outcome depend on network
    # timing. A second "yes" after a "no" is worse.
    identity, sign = device
    gate, _ = await _gate(tmp_path, _Executor())
    record = await _proposed(gate)

    await gate.decide(
        record.id,
        approved=True,
        identity=identity,
        signature=sign(record.id, True, record.nonce),
    )
    again = await gate.decide(
        record.id,
        approved=False,
        identity=identity,
        signature=sign(record.id, False, record.nonce),
    )

    assert again.status is State.APPROVED


async def test_an_answer_after_the_deadline_expires_instead_of_approving(tmp_path, device):
    identity, sign = device
    executor = _Executor()
    gate, _ = await _gate(tmp_path, executor)
    record = await _proposed(gate)

    late = await gate.decide(
        record.id,
        approved=True,
        identity=identity,
        signature=sign(record.id, True, record.nonce),
        now=record.respond_by + timedelta(seconds=1),
    )

    assert late.status is State.EXPIRED
    assert executor.calls == []


async def test_a_lapsed_proposal_is_recorded_as_expired_not_left_pending(tmp_path):
    """A caller waiting on the answer needs one, even when it is 'nobody said'."""

    gate, _ = await _gate(tmp_path, _Executor())
    record = await _proposed(gate)

    lapsed = await gate.expire_lapsed(now=record.respond_by + timedelta(seconds=1))

    assert lapsed == (record.id,)
    current = await gate.get(record.id)
    assert current is not None and current.status is State.EXPIRED


# -- forgery and replay -------------------------------------------------------


async def test_a_forged_signature_is_refused(tmp_path, device):
    identity, _ = device
    executor = _Executor()
    gate, _ = await _gate(tmp_path, executor)
    record = await _proposed(gate)

    with pytest.raises(AuthenticationError):
        await gate.decide(
            record.id,
            approved=True,
            identity=identity,
            signature=base64.b64encode(b"not a signature").decode(),
        )

    current = await gate.get(record.id)
    assert current is not None and current.status is State.PENDING
    assert executor.calls == []


async def test_a_yes_captured_for_one_proposal_cannot_approve_another(tmp_path, device):
    # The nonce is what makes a decision single-use rather than merely
    # authentic: without it, one captured "yes" is a valid "yes" forever.
    identity, sign = device
    gate, _ = await _gate(tmp_path, _Executor())
    first = await _proposed(gate)
    second = await _proposed(gate)

    stolen = sign(first.id, True, first.nonce)

    with pytest.raises(AuthenticationError):
        await gate.decide(
            second.id, approved=True, identity=identity, signature=stolen
        )
    assert first.nonce != second.nonce


async def test_a_signature_for_yes_cannot_be_replayed_as_no(tmp_path, device):
    """The decision itself is inside the signed material, not beside it."""

    identity, sign = device
    gate, _ = await _gate(tmp_path, _Executor())
    record = await _proposed(gate)
    yes = sign(record.id, True, record.nonce)

    with pytest.raises(AuthenticationError):
        await gate.decide(
            record.id, approved=False, identity=identity, signature=yes
        )


async def test_a_session_with_no_certificate_cannot_approve_anything(tmp_path):
    gate, _ = await _gate(tmp_path, _Executor())
    record = await _proposed(gate)
    anonymous = AuthenticatedIdentity(
        identity="companion-1",
        roles=("companion",),
        not_after=datetime.now(UTC) + timedelta(days=1),
        fingerprint="test",
    )

    with pytest.raises(AuthenticationError, match="no certificate"):
        await gate.decide(
            record.id, approved=True, identity=anonymous, signature="AAAA"
        )


# -- execution and restart ----------------------------------------------------


async def test_executing_twice_does_not_perform_the_mutation_twice(tmp_path, device):
    identity, sign = device
    executor = _Executor()
    gate, _ = await _gate(tmp_path, executor)
    record = await _proposed(gate)
    await gate.decide(
        record.id,
        approved=True,
        identity=identity,
        signature=sign(record.id, True, record.nonce),
    )

    await gate.execute_approved(record.id)
    again = await gate.execute_approved(record.id)

    assert again.status is State.EXECUTED
    assert executor.calls == [record.idempotency_key]


async def test_a_restart_mid_execution_leaves_the_approval_visibly_unfinished(
    tmp_path, device
):
    # The ordering that makes recovery possible: a crash between the decision
    # and the write must not look like "never started".
    identity, sign = device
    executor = _Executor(fail=True)
    gate, _ = await _gate(tmp_path, executor)
    record = await _proposed(gate)
    await gate.decide(
        record.id,
        approved=True,
        identity=identity,
        signature=sign(record.id, True, record.nonce),
    )

    failed = await gate.execute_approved(record.id)
    assert failed.status is State.FAILED
    assert failed.failure_detail is not None

    # And a failed mutation is not silently retried by a later call.
    again = await gate.execute_approved(record.id)
    assert again.status is State.FAILED
    assert len(executor.calls) == 1


async def test_recovery_finishes_an_approval_caught_mid_execution(tmp_path, device):
    identity, sign = device
    executor = _Executor()
    gate, store = await _gate(tmp_path, executor)
    record = await _proposed(gate)
    await gate.decide(
        record.id,
        approved=True,
        identity=identity,
        signature=sign(record.id, True, record.nonce),
    )

    # Simulate the crash: claimed for execution, never finished.
    stored = await store.get(CompanionApprovalRecord, record.id)
    assert stored is not None
    await store.save(
        stored.record.model_copy(update={"status": State.EXECUTING}),
        expected_revision=stored.revision,
    )

    resumed = await gate.recover_interrupted()

    assert resumed == (record.id,)
    finished = await gate.get(record.id)
    assert finished is not None and finished.status is State.EXECUTED
    # Retried under the key that was stored before the first attempt, so the
    # executor can collapse it if the first attempt did land.
    assert executor.calls == [record.idempotency_key]


async def test_an_approval_survives_a_server_restart(tmp_path, device):
    identity, sign = device
    executor = _Executor()
    path = tmp_path / "durable.sqlite3"
    store = DurableAgentStore(path)
    await store.initialize()
    gate = CompanionApprovalGate(store, execute=executor)
    record = await _proposed(gate)

    restarted = DurableAgentStore(path)
    await restarted.initialize()
    reopened = CompanionApprovalGate(restarted, execute=executor)

    decided = await reopened.decide(
        record.id,
        approved=True,
        identity=identity,
        signature=sign(record.id, True, record.nonce),
    )
    done = await reopened.execute_approved(record.id)

    assert decided.status is State.APPROVED
    assert done.status is State.EXECUTED
    assert executor.calls == [record.idempotency_key]
