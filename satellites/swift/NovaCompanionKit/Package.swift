// swift-tools-version: 6.0
import PackageDescription

// Wire types for the companion channel, shared by the iOS app and anything
// else that has to speak the protocol. Kept as a library with no Apple-only
// dependencies so `swift test` runs headlessly on the build host — the
// contract can be verified without a device, a simulator, or an app.
let package = Package(
    name: "NovaCompanionKit",
    platforms: [.iOS(.v18), .macOS(.v14)],
    products: [
        .library(name: "NovaCompanionKit", targets: ["NovaCompanionKit"])
    ],
    targets: [
        .target(name: "NovaCompanionKit"),
        .testTarget(name: "NovaCompanionKitTests", dependencies: ["NovaCompanionKit"]),
    ]
)
