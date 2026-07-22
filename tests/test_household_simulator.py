from __future__ import annotations

from datetime import UTC, datetime

from nova_voice.capabilities.registry import CapabilityRegistry
from nova_voice.domain import CapabilityToolCall, PlannedAction
from nova_voice.evaluation.household import (
    HouseholdSimulator,
    SimulatedClock,
    SimulatedHouseholdProvider,
)


def _action(tool: str, **arguments) -> PlannedAction:
    return PlannedAction(
        id=f"action-{tool}",
        order=0,
        call=CapabilityToolCall(
            provider="sim_household",
            tool=tool,
            arguments=arguments,
        ),
    )


async def test_provider_models_delayed_state_convergence_and_failures() -> None:
    simulator = HouseholdSimulator(
        clock=SimulatedClock(datetime(2026, 7, 22, 9, 0, tzinfo=UTC))
    )
    simulator.add_entity("light.office", "off", room="office")
    provider = SimulatedHouseholdProvider(simulator)
    registry = CapabilityRegistry(allowlist={"sim_household"})
    registry.register(provider)

    action = registry.validate_action(
        _action(
            "sim.household.set",
            entityId="light.office",
            state="on",
            delaySeconds=5,
        )
    )
    scheduled = await provider.execute(action)

    assert scheduled.ok
    assert scheduled.observed["entities"]["light.office"]["state"] == "off"
    simulator.advance(4.9)
    assert simulator.snapshot()["entities"]["light.office"]["state"] == "off"
    simulator.advance(0.1)
    assert simulator.snapshot()["entities"]["light.office"]["state"] == "on"

    simulator.fail_next("sim.household.set", message="radio unavailable")
    failed = await provider.execute(action)
    recovered = await provider.execute(action)
    assert (failed.ok, failed.code, failed.message) == (
        False,
        "backend_error",
        "radio unavailable",
    )
    assert recovered.ok


def test_fake_clock_orders_occupancy_and_concurrent_speakers_repeatably() -> None:
    def run_once() -> tuple[dict, list[dict]]:
        simulator = HouseholdSimulator()
        simulator.set_occupancy("Addie", "office", delay_seconds=2)
        simulator.speak("Addie", "office", "Nova, turn it on", delay_seconds=3)
        simulator.speak("Guest", "office", "I was talking", delay_seconds=3)

        first = simulator.advance(2)
        second = simulator.advance(1)

        assert first[0]["kind"] == "occupancy"
        assert [event["speaker"] for event in second] == ["Addie", "Guest"]
        assert second[0]["at"] == second[1]["at"]
        return simulator.snapshot(), simulator.event_log

    first_snapshot, first_log = run_once()
    second_snapshot, second_log = run_once()

    assert first_snapshot == second_snapshot
    assert first_log == second_log
    assert first_snapshot["occupancy"] == {"Addie": "office"}


async def test_simulator_query_and_health_use_controlled_time() -> None:
    simulator = HouseholdSimulator()
    provider = SimulatedHouseholdProvider(simulator)
    simulator.advance(90)

    result = await provider.execute(_action("sim.household.query"))
    health = await provider.health()

    assert result.ok
    assert result.observed["at"] == "2026-01-01T00:01:30+00:00"
    assert health == {"ok": True, "clock": "2026-01-01T00:01:30+00:00"}
