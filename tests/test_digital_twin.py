from nova_voice.digital_twin import DigitalTwinProvider, HouseholdDigitalTwin
from nova_voice.domain import CapabilityToolCall, PlannedAction


def _state() -> dict:
    return {
        "generatedAt": "revision-1",
        "eventKinds": ["energy"],
        "entities": {
            "light.office": {
                "state": "on",
                "attributes": {"power_w": 10},
            },
            "heater.lounge": {
                "state": "off",
                "attributes": {"power_w": 2000},
            },
        },
    }


def test_digital_twin_simulates_causal_and_energy_delta_without_mutation() -> None:
    state = _state()
    scenario = HouseholdDigitalTwin().simulate(
        state,
        [
            {
                "entityId": "heater.lounge",
                "state": "on",
                "cause": "the lounge drops below 18 degrees",
            }
        ],
        duration_hours=2,
    )

    assert scenario.side_effects == 0
    assert scenario.baseline_watts == 10
    assert scenario.projected_watts == 2010
    assert scenario.projected_energy_kwh == 4.02
    assert "drops below 18 degrees" in scenario.explanations[0]
    assert state["entities"]["heater.lounge"]["state"] == "off"


def test_digital_twin_warns_instead_of_inventing_unknown_entity() -> None:
    scenario = HouseholdDigitalTwin().simulate(
        _state(), [{"entityId": "switch.missing", "state": "on"}]
    )

    assert scenario.changes == ()
    assert scenario.warnings == ("unknown entity: switch.missing",)


def test_automation_rehearsal_is_side_effect_free_and_trigger_aware() -> None:
    result = HouseholdDigitalTwin().rehearse_automation(
        _state(),
        {
            "trigger": {"kind": "energy"},
            "actions": [{"channel": "notification", "message": "High use"}],
        },
    )

    assert result["triggerMatched"] is True
    assert result["proposedActionCount"] == 1
    assert result["sideEffects"] == 0


async def test_provider_exposes_read_only_what_if_tools() -> None:
    async def state_supplier() -> dict:
        return _state()

    provider = DigitalTwinProvider(state_supplier)
    action = PlannedAction(
        id="what-if",
        order=0,
        call=CapabilityToolCall(
            provider="household_digital_twin",
            tool="twin.energy",
            arguments={
                "actions": [{"entityId": "heater.lounge", "state": "on"}],
                "durationHours": 1,
            },
        ),
    )

    result = await provider.execute(action)

    assert result.ok
    assert result.observed is not None
    assert result.observed["side_effects"] == 0
    policies = provider.manifest().tool_policies
    assert all(policy.cancellation == "anytime" for policy in policies.values())
