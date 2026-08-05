import Foundation
import Hummingbird
import MLXGateway

@main
struct GatewayServer {
    static func main() async throws {
        let router = Router()

        // API key auth — enabled when API_KEY_FILE env var points to a key file.
        if let keyFilePath = ProcessInfo.processInfo.environment["API_KEY_FILE"] {
            let auth = try BearerAuthMiddleware<BasicRequestContext>.loadFromFile(at: keyFilePath)
            router.add(middleware: auth)
        }

        // Per-request deadline — configurable via REQUEST_TIMEOUT_SECONDS (default 120 s).
        let timeoutSeconds: Double
        if let raw = ProcessInfo.processInfo.environment["REQUEST_TIMEOUT_SECONDS"],
           let parsed = Double(raw) {
            timeoutSeconds = parsed
        } else {
            timeoutSeconds = 120
        }

        // Max concurrently-loaded models — configurable via MAX_MODELS (default 3).
        let maxModels: Int
        if let raw = ProcessInfo.processInfo.environment["MAX_MODELS"],
           let parsed = Int(raw) {
            maxModels = parsed
        } else {
            maxModels = 3
        }

        let engine = MLXInferenceEngine(maxModels: maxModels)
        let assembler = BatchAssembler(
            configuration: BatchAssemblerConfig(maxBatchSize: 8, maxWaitMs: 100),
            handler: engine.makeBatchHandler()
        )

        let chatRouter = ChatRouter(
            handler: { request in
                try await assembler.submit(request)
            },
            timeoutSeconds: timeoutSeconds
        )

        chatRouter.addRoutes(to: router)

        // GET /v1/models — list models currently loaded in the pool.
        // Only models that have been requested at least once appear here;
        // the engine loads from HuggingFace on demand, not from a local directory.
        router.get("/v1/models") { _, _ async -> Response in
            let ids = await engine.loadedModelIDs()
            let list = ModelListResponse(data: ids.map { ModelInfo(id: $0) })
            let data = (try? JSONEncoder().encode(list)) ?? Data()
            return Response(
                status: .ok,
                headers: [.contentType: "application/json"],
                body: .init(byteBuffer: .init(bytes: data))
            )
        }

        // GET /health — liveness probe.
        router.get("/health") { _, _ in
            Response(
                status: .ok,
                headers: [.contentType: "application/json"],
                body: .init(byteBuffer: .init(string: #"{"status":"ok"}"#))
            )
        }

        let app = Application(
            router: router,
            configuration: .init(address: .hostname("127.0.0.1", port: 8080))
        )

        try await app.run()
    }
}
