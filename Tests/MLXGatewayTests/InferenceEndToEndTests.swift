import XCTest
import Hummingbird
import HummingbirdTesting
import NIOCore
@testable import MLXGateway

/// End-to-end inference test.
///
/// Sends a real `/v1/chat/completions` request through the full stack:
///   ChatRouter → BatchAssembler → MLXInferenceEngine → MLX model → response
///
/// Requires `mlx-community/Qwen2.5-3B-bf16` to be present in the local
/// HuggingFace cache (`~/.cache/huggingface/hub/`).
///
/// Run with:
///   swift test --filter InferenceEndToEndTests
final class InferenceEndToEndTests: XCTestCase {

    private static let modelID = "mlx-community/Qwen2.5-3B-bf16"

    func testChatCompletionReturnsCoherentResponse() async throws {
        let engine = MLXInferenceEngine()
        let assembler = BatchAssembler(
            configuration: BatchAssemblerConfig(maxBatchSize: 4, maxWaitMs: 50),
            handler: engine.makeBatchHandler()
        )

        let router = Router()
        let chatRouter = ChatRouter { request in
            try await assembler.submit(request)
        }
        chatRouter.addRoutes(to: router)

        let app = Application(router: router)

        let request = ChatCompletionRequest(
            model: Self.modelID,
            messages: [ChatMessage(role: .user, content: "What is 2 + 2? Reply with just the number.")],
            maxTokens: 16
        )
        let requestBytes = try JSONEncoder().encode(request)
        let requestBuffer = ByteBuffer(bytes: Array(requestBytes))

        try await app.test(.live) { client in
            let response = try await client.execute(
                uri: "/v1/chat/completions",
                method: .post,
                headers: [.contentType: "application/json"],
                body: requestBuffer
            )

            XCTAssertEqual(response.status, .ok, "Expected HTTP 200")

            var body = response.body
            let responseData = Data(body.readBytes(length: body.readableBytes) ?? [])
            let completion = try JSONDecoder().decode(ChatCompletionResponse.self, from: responseData)

            XCTAssertEqual(completion.model, Self.modelID)
            XCTAssertFalse(completion.choices.isEmpty, "Expected at least one choice")

            guard let firstChoice = completion.choices.first else {
                XCTFail("No choices in response")
                return
            }
            XCTAssertEqual(firstChoice.message.role, .assistant)

            let text = firstChoice.message.content.textContent
            XCTAssertFalse(text.isEmpty, "Model produced no output text")
            print("Model response: \"\(text)\"")
        }
    }

    func testBatchHandlerDirectly() async throws {
        let engine = MLXInferenceEngine()
        let handler = engine.makeBatchHandler()

        let requests = [
            ChatCompletionRequest(
                model: Self.modelID,
                messages: [ChatMessage(role: .user, content: "Say the word 'hello'.")],
                maxTokens: 10
            )
        ]

        let responses = try await handler(requests)

        XCTAssertEqual(responses.count, 1)
        let text = responses[0].choices.first?.message.content.textContent ?? ""
        XCTAssertFalse(text.isEmpty, "Direct batch handler produced no output")
        print("Direct handler response: \"\(text)\"")
    }
}
