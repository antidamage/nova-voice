import CoreLocation
import Foundation
import NovaCompanionKit

/// Answers "is this phone at home", and nothing more precise than that.
///
/// The home coordinate is configured on the device and never leaves it. What
/// crosses to Nova is a boolean, an age and a precision — enough to decide
/// whether to act on it, and not enough to locate a household.
///
/// Note the deliberate asymmetry with the rest of the companion: everything
/// else answers *bounded* questions and lets Nova reason over the result. This
/// one answers a *derived* question, because the raw value is the one piece of
/// personal context where the derived form loses nothing Nova needed and the
/// raw form is uniquely sensitive.
@MainActor
final class HomeLocationRuntime: NSObject, CLLocationManagerDelegate {
    struct Fix {
        let isHome: Bool
        let ageSeconds: Double
    }

    /// Inside this distance counts as home. Wide enough to cover a section and
    /// GPS drift indoors, narrow enough that the next street does not.
    private let homeRadius: CLLocationDistance
    private let home: CLLocation?
    private let manager = CLLocationManager()
    private var pending: [CheckedContinuation<CLLocation?, Never>] = []

    init(home: CLLocationCoordinate2D?, radiusMetres: CLLocationDistance = 120) {
        self.home = home.map { CLLocation(latitude: $0.latitude, longitude: $0.longitude) }
        self.homeRadius = radiusMetres
        super.init()
        manager.delegate = self
    }

    var permission: PersonalPermission {
        switch manager.authorizationStatus {
        case .notDetermined: .notDetermined
        case .restricted: .restricted
        case .denied: .denied
        // "When in use" is the honest level for this: the app answers location
        // questions while it is running, and asking for Always would request a
        // capability it does not need to do its job.
        case .authorizedWhenInUse, .authorizedAlways: .authorised
        @unknown default: .restricted
        }
    }

    func requestAccess() {
        guard permission.worthRequesting else { return }
        manager.requestWhenInUseAuthorization()
    }

    func reading(fine: Bool, maxAge: TimeInterval) async -> Fix? {
        guard permission.usable, let home else { return nil }
        manager.desiredAccuracy =
            fine ? kCLLocationAccuracyNearestTenMeters : kCLLocationAccuracyHundredMeters

        // The cached fix first. Asking for a fresh one costs battery and takes
        // seconds, and for "am I home" a two-minute-old fix is almost always
        // the same answer as a new one.
        if let cached = manager.location, cached.timestamp.timeIntervalSinceNow > -maxAge {
            return fix(from: cached, home: home)
        }

        let fresh: CLLocation? = await withCheckedContinuation { continuation in
            pending.append(continuation)
            manager.requestLocation()
        }
        guard let fresh else { return nil }
        return fix(from: fresh, home: home)
    }

    private func fix(from location: CLLocation, home: CLLocation) -> Fix {
        Fix(
            isHome: location.distance(from: home) <= homeRadius,
            ageSeconds: max(0, -location.timestamp.timeIntervalSinceNow)
        )
    }

    private func resolve(_ location: CLLocation?) {
        let waiting = pending
        pending.removeAll()
        for continuation in waiting { continuation.resume(returning: location) }
    }

    nonisolated func locationManager(
        _ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]
    ) {
        let latest = locations.last
        Task { @MainActor in self.resolve(latest) }
    }

    nonisolated func locationManager(_ manager: CLLocationManager, didFailWithError error: Error) {
        // Failure resolves the waiters with nothing rather than leaving them
        // parked: Nova is waiting on this call with a deadline, and silence
        // costs it the whole deadline to learn the same thing.
        Task { @MainActor in self.resolve(nil) }
    }
}
