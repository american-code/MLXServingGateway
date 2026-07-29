// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "GatewayServer",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .library(name: "MLXGateway", targets: ["MLXGateway"]),
        .executable(name: "GatewayServer", targets: ["GatewayServer"]),
    ],
    dependencies: [
        .package(url: "https://github.com/hummingbird-project/hummingbird.git", from: "2.0.0"),
    ],
    targets: [
        .target(
            name: "MLXGateway",
            dependencies: [
                .product(name: "Hummingbird", package: "hummingbird"),
            ]
        ),
        .executableTarget(
            name: "GatewayServer",
            dependencies: [
                "MLXGateway",
                .product(name: "Hummingbird", package: "hummingbird"),
            ]
        ),
    ]
)
