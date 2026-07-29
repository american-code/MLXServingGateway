import Foundation
import Hummingbird
import MLXGateway

@main
struct GatewayServer {
    static func main() async throws {
        let router = Router()

        let chatRouter = ChatRouter { request in
            // Placeholder: replace with actual MLX model inference
            ChatCompletionResponse(
                id: "chatcmpl-\(UUID().uuidString)",
                created: Int(Date().timeIntervalSince1970),
                model: request.model,
                choices: [
                    Choice(
                        index: 0,
                        message: ChatMessage(role: .assistant, content: "Hello from MLX Gateway!"),
                        finishReason: .stop
                    )
                ],
                usage: Usage(promptTokens: 0, completionTokens: 0)
            )
        }

        chatRouter.addRoutes(to: router)

        let app = Application(
            router: router,
            configuration: .init(address: .hostname("127.0.0.1", port: 8080))
        )

        try await app.run()
    }
}
