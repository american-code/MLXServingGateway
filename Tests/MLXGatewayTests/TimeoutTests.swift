import XCTest
import Hummingbird
import HummingbirdTesting
@testable import MLXGateway

// MARK: - Fixtures

private func okResponse(for request: ChatCompletionRequest) -> ChatCompletionResponse {
    ChatCompletionResponse(
        id: "chatcmpl-test",
        created: 0,
        model: request.model,
        choices: [Choice(
            index: 0,
            message: ChatMessage(role: .assistant, content: "pong"),
            finishReason: .stop
        )],
        usage: Usage(promptTokens: 1, completionTokens: 1)
    )
}

private func requestBuffer(stream: Bool = false) throws -> ByteBuffer {
    let req = ChatCompletionRequest(
        model: "test-model",
        messages: [ChatMessage(role: .user, content: "ping")],
        stream: stream ? true : nil
    )
    return ByteBuffer(bytes: try JSONEncoder().encode(req))
}

private func makeRouter(chatRouter: ChatRouter) -> Router<BasicRequestContext> {
    let router = Router()
    chatRouter.addRoutes(to: router)
    return router
}

// MARK: - Tests

final class TimeoutTests: XCTestCase {

    // MARK: Handler completes before deadline → 200

    func testFastHandler_returns200() async throws {
        let chatRouter = ChatRouter(
            handler: { req in okResponse(for: req) },
            timeoutSeconds: 5.0
        )
        let app = Application(router: makeRouter(chatRouter: chatRouter))

        try await app.test(.router) { client in
            let resp = try await client.execute(
                uri: "/v1/chat/completions",
                method: .post,
                headers: [.contentType: "application/json"],
                body: try requestBuffer()
            )
            XCTAssertEqual(resp.status, .ok,
                           "Handler completing within deadline must return 200")
        }
    }

    // MARK: Handler exceeds deadline → 504

    func testSlowHandler_returns504() async throws {
        let chatRouter = ChatRouter(
            handler: { _ in
                try await Task.sleep(nanoseconds: 10_000_000_000)  // 10 s
                fatalError("unreachable")
            },
            timeoutSeconds: 0.05   // 50 ms deadline
        )
        let app = Application(router: makeRouter(chatRouter: chatRouter))

        try await app.test(.router) { client in
            let resp = try await client.execute(
                uri: "/v1/chat/completions",
                method: .post,
                headers: [.contentType: "application/json"],
                body: try requestBuffer()
            )
            XCTAssertEqual(resp.status, .gatewayTimeout,
                           "Handler exceeding deadline must return 504 Gateway Timeout")
            XCTAssertEqual(resp.headers[.contentType], "application/json",
                           "504 response must be JSON")
            let body = String(buffer: resp.body)
            XCTAssertTrue(body.contains("error"), "504 body must carry an error object")
        }
    }

    // MARK: No timeout configured — slow handler still completes

    func testNoTimeout_slowHandlerCompletes() async throws {
        let chatRouter = ChatRouter(
            handler: { req in
                try await Task.sleep(nanoseconds: 20_000_000)  // 20 ms
                return okResponse(for: req)
            },
            timeoutSeconds: nil
        )
        let app = Application(router: makeRouter(chatRouter: chatRouter))

        try await app.test(.router) { client in
            let resp = try await client.execute(
                uri: "/v1/chat/completions",
                method: .post,
                headers: [.contentType: "application/json"],
                body: try requestBuffer()
            )
            XCTAssertEqual(resp.status, .ok,
                           "Without a configured timeout, slow handlers must still return 200")
        }
    }

    // MARK: Streaming handler exceeds deadline → 504

    func testSlowStreamHandler_returns504() async throws {
        let chatRouter = ChatRouter(
            handler: { req in okResponse(for: req) },
            streamHandler: { _ in
                try await Task.sleep(nanoseconds: 10_000_000_000)  // 10 s
                fatalError("unreachable")
            },
            timeoutSeconds: 0.05
        )
        let app = Application(router: makeRouter(chatRouter: chatRouter))

        try await app.test(.router) { client in
            let resp = try await client.execute(
                uri: "/v1/chat/completions",
                method: .post,
                headers: [.contentType: "application/json"],
                body: try requestBuffer(stream: true)
            )
            XCTAssertEqual(resp.status, .gatewayTimeout,
                           "Streaming handler exceeding deadline must return 504")
        }
    }

    // MARK: 504 body is valid APIError JSON

    func testTimeout_responseBodyIsValidAPIError() async throws {
        let chatRouter = ChatRouter(
            handler: { _ in
                try await Task.sleep(nanoseconds: 10_000_000_000)
                fatalError("unreachable")
            },
            timeoutSeconds: 0.05
        )
        let app = Application(router: makeRouter(chatRouter: chatRouter))

        try await app.test(.router) { client in
            let resp = try await client.execute(
                uri: "/v1/chat/completions",
                method: .post,
                headers: [.contentType: "application/json"],
                body: try requestBuffer()
            )
            XCTAssertEqual(resp.status, .gatewayTimeout)
            let data = Data(String(buffer: resp.body).utf8)
            let apiError = try JSONDecoder().decode(APIError.self, from: data)
            XCTAssertFalse(apiError.error.message.isEmpty,
                           "APIError message must not be empty")
        }
    }
}
