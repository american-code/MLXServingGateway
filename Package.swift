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
        .package(url: "https://github.com/ml-explore/mlx-swift-lm.git", from: "3.31.3"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "0.1.0"),
        .package(url: "https://github.com/huggingface/swift-huggingface", from: "0.9.0"),
    ],
    targets: [
        .target(
            name: "MLXGateway",
            dependencies: [
                .product(name: "Hummingbird", package: "hummingbird"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
                .product(name: "HuggingFace", package: "swift-huggingface"),
                .product(name: "Transformers", package: "swift-transformers"),
            ]
        ),
        .executableTarget(
            name: "GatewayServer",
            dependencies: [
                "MLXGateway",
                .product(name: "Hummingbird", package: "hummingbird"),
            ]
        ),
        .testTarget(
            name: "MLXGatewayTests",
            dependencies: [
                "MLXGateway",
                .product(name: "Hummingbird", package: "hummingbird"),
                .product(name: "HummingbirdTesting", package: "hummingbird"),
            ]
        ),
        .executableTarget(
            name: "KVCacheBenchmark",
            dependencies: []
        ),
    ]
)
