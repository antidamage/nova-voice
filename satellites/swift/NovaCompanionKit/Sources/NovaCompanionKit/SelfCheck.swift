import Foundation

/// Whether this device should take a job *right now*.
///
/// Iridium already gates on telemetry, so this is not a second opinion about
/// the same facts — it is a fresher one. Telemetry is sent on a cycle and on
/// change, but an offer can still arrive in the gap: the phone reports nominal
/// thermals, goes into a pocket in direct sun, and is offered work a moment
/// later against a reading that is now wrong.
///
/// Checking here costs the server one round trip. Not checking costs it the
/// job's entire completion deadline before it falls back, because a device
/// that accepts and then struggles holds the attempt until it times out. That
/// asymmetry is the whole argument for saying no early and saying it precisely.
public struct CompanionSelfCheck: Sendable {
    /// Below this, a job is declined unless the device is on mains power.
    /// Matches the server's `off_below` floor, deliberately: two components
    /// disagreeing about what "too flat to help" means would produce a device
    /// that is offered work it always refuses.
    public var batteryFloor: Double
    /// Interactive work is declined at `serious`; everything is declined at
    /// `critical`. `serious` already means the OS is throttling, so a job
    /// accepted there will be slow *and* make the throttling worse.
    public var refuseAtSeriousThermal: Bool

    public init(batteryFloor: Double = 0.15, refuseAtSeriousThermal: Bool = true) {
        self.batteryFloor = batteryFloor
        self.refuseAtSeriousThermal = refuseAtSeriousThermal
    }

    /// A reason to decline, or nil to go ahead.
    ///
    /// Order matters: the most specific and least recoverable reason wins, so
    /// the server's failure classification and the owner's status screen name
    /// the thing that actually stopped it rather than the first thing checked.
    public func refusal(
        for telemetry: CompanionTelemetry,
        workload: CompanionWorkload
    ) -> (reason: RejectReason, detail: String, retryAfter: Double?)? {
        if !telemetry.models.hotAvailable {
            return (.modelUnavailable, "no on-device model is available", nil)
        }
        if telemetry.thermalState == .critical {
            // No retry hint: the OS decides when this clears, and guessing
            // would have the server back off for the wrong length of time.
            return (.thermal, "thermal state is critical", nil)
        }
        if refuseAtSeriousThermal && telemetry.thermalState == .serious {
            return (.thermal, "thermal state is serious", 120)
        }
        if !telemetry.charging && telemetry.battery < batteryFloor {
            return (
                .battery,
                "battery is below the floor for taking work",
                // Only worth re-offering once something has charged.
                600
            )
        }
        if telemetry.lowPowerMode && !telemetry.charging {
            // Low Power Mode is the owner asking for the battery to last. A
            // background classification is not worth overriding that; the
            // server will run it locally at no cost to this device.
            return (.battery, "low power mode is on", 300)
        }
        return nil
    }
}

extension CompanionTelemetry {
    /// True when the device is in worse shape than when the job was accepted.
    ///
    /// Used mid-execution: a long job should give up on a phone that has
    /// started overheating rather than finish and make it worse.
    public func degraded(from earlier: CompanionTelemetry) -> Bool {
        if thermalState.severity > earlier.thermalState.severity { return true }
        if !charging && earlier.charging { return true }
        if lowPowerMode && !earlier.lowPowerMode { return true }
        return false
    }
}

extension ThermalState {
    var severity: Int {
        switch self {
        case .nominal: 0
        case .fair: 1
        case .serious: 2
        case .critical: 3
        }
    }
}
