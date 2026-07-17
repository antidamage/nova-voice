// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "NovaVoiceSatellite",
    platforms: [.macOS(.v14)],
    products: [
        .executable(name: "NovaVoiceSatellite", targets: ["NovaVoiceSatellite"])
    ],
    targets: [
        .target(name: "NVSShim"),
        .executableTarget(
            name: "NovaVoiceSatellite",
            dependencies: ["NVSShim"],
            swiftSettings: [.swiftLanguageMode(.v5)]
        )
    ]
)
