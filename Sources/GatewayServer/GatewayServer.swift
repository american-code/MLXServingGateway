import Foundation
import Hummingbird
import MLXGateway

@main
struct GatewayServer {
    static func main() async throws {
        let router = Router()

        let engine = MLXInferenceEngine()
        let assembler = BatchAssembler(
            configuration: BatchAssemblerConfig(maxBatchSize: 8, maxWaitMs: 100),
            handler: engine.makeBatchHandler()
        )

        let chatRouter = ChatRouter { request in
            try await assembler.submit(request)
        }

        chatRouter.addRoutes(to: router)

        let app = Application(
            router: router,
            configuration: .init(address: .hostname("127.0.0.1", port: 8080))
        )

        try await app.run()
    }
}
