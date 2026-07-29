import Foundation
import Hummingbird
import HummingbirdTesting
import XCTest
@testable import MLXGateway

final class StreamingTests: XCTestCase {

    // MARK: - Fixtures

    private let tokens = ["Hello", ",", " ", "world", "!"]

    private var expectedContent: String { tokens.joined() }

    private func makeRouter() -> Router<BasicRequestContext> {
        let router = Router()
        let tokens = self.tokens
        let expectedContent = self.expectedContent

        let chatRouter = ChatRouter(
            handler: { request in
                ChatCompletionResponse(
                    id: "chatcmpl-test",
                    created: 0,
                    model: request.model,
                    choices: [Choice(
                        index: 0,
                        message: ChatMessage(role: .assistant, content: expectedContent),
                        finishReason: .stop
                    )],
                    usage: Usage(promptTokens: 5, completionTokens: tokens.count)
                )
            },
            streamHandler: { _ in
                AsyncThrowingStream { continuation in
                    for token in tokens {
                        continuation.yield(token)
                    }
                    continuation.finish()
                }
            }
        )

        chatRouter.addRoutes(to: router)
        return router
    }

    private func requestBuffer(stream: Bool) throws -> ByteBuffer {
        let req = ChatCompletionRequest(
            model: "test-model",
            messages: [ChatMessage(role: .user, content: "ping")],
            stream: stream
        )
        return ByteBuffer(bytes: try JSONEncoder().encode(req))
    }

    // MARK: - Core parity test

    func testConcatenatedStreamTokensMatchNonStreamingContent() async throws {
        let app = Application(router: makeRouter())
        let nonStreamBuffer = try requestBuffer(stream: false)
        let streamBuffer = try requestBuffer(stream: true)

        try await app.test(.router) { client in
            // Non-streaming: get the full content string
            let nonStreamResp = try await client.execute(
                uri: "/v1/chat/completions",
                method: .post,
                headers: [.contentType: "application/json"],
                body: nonStreamBuffer
            )
            XCTAssertEqual(nonStreamResp.status, .ok)
            XCTAssertEqual(nonStreamResp.headers[.contentType], "application/json")

            let completion = try JSONDecoder().decode(
                ChatCompletionResponse.self,
                from: Data(String(buffer: nonStreamResp.body).utf8)
            )
            guard case .text(let nonStreamContent) = completion.choices.first?.message.content else {
                return XCTFail("Expected text content in non-streaming response")
            }

            // Streaming: collect and concatenate delta tokens
            let streamResp = try await client.execute(
                uri: "/v1/chat/completions",
                method: .post,
                headers: [.contentType: "application/json"],
                body: streamBuffer
            )
            XCTAssertEqual(streamResp.status, .ok)
            XCTAssertEqual(streamResp.headers[.contentType], "text/event-stream")

            let concatenated = try collectSSEContent(String(buffer: streamResp.body))
            XCTAssertEqual(concatenated, nonStreamContent,
                           "Streaming token stream must reconstruct the non-streaming content")
        }
    }

    // MARK: - First chunk carries role

    func testFirstChunkCarriesAssistantRole() async throws {
        let app = Application(router: makeRouter())
        let body = try requestBuffer(stream: true)

        try await app.test(.router) { client in
            let resp = try await client.execute(
                uri: "/v1/chat/completions",
                method: .post,
                headers: [.contentType: "application/json"],
                body: body
            )
            let events = sseEventPayloads(String(buffer: resp.body))
            guard let firstPayload = events.first, firstPayload != "[DONE]" else {
                return XCTFail("No SSE events received")
            }
            let chunk = try JSONDecoder().decode(
                ChatCompletionChunk.self,
                from: Data(firstPayload.utf8)
            )
            XCTAssertEqual(chunk.choices.first?.delta.role, .assistant,
                           "First SSE chunk must carry role=assistant")
            XCTAssertNil(chunk.choices.first?.delta.content,
                         "First SSE chunk must not carry content")
        }
    }

    // MARK: - Final chunk has finish_reason=stop and [DONE] sentinel

    func testFinalChunkHasFinishReasonStopAndDoneSentinel() async throws {
        let app = Application(router: makeRouter())
        let body = try requestBuffer(stream: true)

        try await app.test(.router) { client in
            let resp = try await client.execute(
                uri: "/v1/chat/completions",
                method: .post,
                headers: [.contentType: "application/json"],
                body: body
            )
            let events = sseEventPayloads(String(buffer: resp.body))
            XCTAssertTrue(events.contains("[DONE]"), "SSE stream must end with [DONE]")

            let chunkPayloads = events.filter { $0 != "[DONE]" }
            guard let lastPayload = chunkPayloads.last else {
                return XCTFail("No chunk events received")
            }
            let lastChunk = try JSONDecoder().decode(
                ChatCompletionChunk.self,
                from: Data(lastPayload.utf8)
            )
            XCTAssertEqual(lastChunk.choices.first?.finishReason, .stop,
                           "Last SSE chunk must have finish_reason=stop")
        }
    }

    // MARK: - Falls back to JSON when streamHandler is nil

    func testFallsBackToJSONWhenNoStreamHandler() async throws {
        let router = Router()
        let chatRouter = ChatRouter(handler: { request in
            ChatCompletionResponse(
                id: "chatcmpl-fallback",
                created: 0,
                model: request.model,
                choices: [Choice(
                    index: 0,
                    message: ChatMessage(role: .assistant, content: "fallback"),
                    finishReason: .stop
                )],
                usage: Usage(promptTokens: 1, completionTokens: 1)
            )
        })
        chatRouter.addRoutes(to: router)
        let app = Application(router: router)
        let body = try requestBuffer(stream: true)

        try await app.test(.router) { client in
            let resp = try await client.execute(
                uri: "/v1/chat/completions",
                method: .post,
                headers: [.contentType: "application/json"],
                body: body
            )
            XCTAssertEqual(resp.status, .ok)
            XCTAssertEqual(resp.headers[.contentType], "application/json")
        }
    }
}

// MARK: - SSE parsing helpers

/// Returns the payload string for each `data:` line, split on blank lines.
private func sseEventPayloads(_ text: String) -> [String] {
    text.components(separatedBy: "\n\n")
        .compactMap { block -> String? in
            let trimmed = block.trimmingCharacters(in: .whitespacesAndNewlines)
            guard trimmed.hasPrefix("data: ") else { return nil }
            return String(trimmed.dropFirst("data: ".count))
        }
}

/// Decodes each non-[DONE] SSE payload as ChatCompletionChunk and joins delta content.
private func collectSSEContent(_ text: String) throws -> String {
    try sseEventPayloads(text)
        .filter { $0 != "[DONE]" }
        .reduce(into: "") { result, payload in
            let chunk = try JSONDecoder().decode(
                ChatCompletionChunk.self,
                from: Data(payload.utf8)
            )
            if let token = chunk.choices.first?.delta.content {
                result += token
            }
        }
}
