import Foundation

/// Which address to try, and in what order.
///
/// This decides more than it looks like it does. The server classifies a
/// session's locality from the **peer address it actually arrives on**, and
/// home-LAN-only routes are the ones that let the phone replace Iridium's
/// reasoning. So a device that reaches the tailnet address first — while
/// sitting on the home Wi-Fi — is correctly classified `tailnet` and correctly
/// refused those routes. It would look like a routing bug and be an ordering
/// bug.
///
/// Lives here rather than in the app because it is a rule with cases, and the
/// app target has nothing that can test it.
public enum CompanionEndpoints {
    /// The LAN address first, always, then the tailnet one as a fallback.
    ///
    /// Not "whichever answers first": a race would make locality depend on
    /// network timing, so the same phone in the same room would sometimes get
    /// home routes and sometimes not.
    public static func ordered(lan: URL?, tailnet: URL?) -> [URL] {
        [lan, tailnet].compactMap { $0 }
    }

    /// Where to resume after a session ends cleanly.
    ///
    /// Back to the start of the list rather than on to the next address. A
    /// session that ran and ended is not evidence the address was bad — the
    /// server restarted, or the app was backgrounded — and carrying on down
    /// the list would quietly demote a phone to `tailnet` for the rest of the
    /// day over one ordinary reconnect.
    public static func afterCleanDisconnect(lan: URL?, tailnet: URL?) -> [URL] {
        ordered(lan: lan, tailnet: tailnet)
    }

    /// True when this address is the one that can unlock home-LAN routes.
    ///
    /// Advisory only, and deliberately so: the device's belief about where it
    /// is never decides anything. The server checks the peer address itself,
    /// and a phone claiming to be home cannot make it so.
    public static func isHomeCandidate(_ endpoint: URL, lan: URL?) -> Bool {
        guard let lan else { return false }
        return endpoint == lan
    }
}
