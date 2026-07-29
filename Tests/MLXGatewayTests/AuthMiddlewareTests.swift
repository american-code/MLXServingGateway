import XCTest
import Hummingbird
import HummingbirdTesting
import Foundation
@testable import MLXGateway

// MARK: - Fixture

private let validKey = "sk-test-valid-key-abc123"
private let anotherValidKey = "sk-test-second-key-xyz"

private func makeRouter(keys: Set<String>) -> Router<BasicRequestContext> {
    let router = Router()
    router.add(middleware: BearerAuthMiddleware(keys: keys))

    let chatRouter = ChatRouter { _ in
        ChatCompletionResponse(
            id: "chatcmpl-test",
            created: 0,
            model: "test-model",
            choices: [Choice(
                index: 0,
                message: ChatMessage(role: .assistant, content: "ok"),
                finishReason: .stop
            )],
            usage: Usage(promptTokens: 1, completionTokens: 1)
        )
    }
    chatRouter.addRoutes(to: router)
    return router
}

private func requestBuffer() throws -> ByteBuffer {
    let req = ChatCompletionRequest(
        model: "test-model",
        messages: [ChatMessage(role: .user, content: "hi")]
    )
    return ByteBuffer(bytes: try JSONEncoder().encode(req))
}

// MARK: - Tests

final class AuthMiddlewareTests: XCTestCase {

    // MARK: Missing header → 401

    func testMissingAuthHeader_returns401() async throws {
        let app = Application(router: makeRouter(keys: [validKey]))

        try await app.test(.router) { client in
            let resp = try await client.execute(
                uri: "/v1/chat/completions",
                method: .post,
                headers: [.contentType: "application/json"],
                body: try requestBuffer()
            )
            XCTAssertEqual(resp.status, .unauthorized, "No Authorization header must return 401")
            assertJSONError(resp)
        }
    }

    // MARK: Wrong key → 401

    func testWrongKey_returns401() async throws {
        let app = Application(router: makeRouter(keys: [validKey]))

        try await app.test(.router) { client in
            let resp = try await client.execute(
                uri: "/v1/chat/completions",
                method: .post,
                headers: [.contentType: "application/json", .authorization: "Bearer wrong-key"],
                body: try requestBuffer()
            )
            XCTAssertEqual(resp.status, .unauthorized, "Invalid API key must return 401")
            assertJSONError(resp)
        }
    }

    // MARK: Malformed header (no "Bearer " prefix) → 401

    func testMalformedHeader_returns401() async throws {
        let app = Application(router: makeRouter(keys: [validKey]))

        try await app.test(.router) { client in
            let resp = try await client.execute(
                uri: "/v1/chat/completions",
                method: .post,
                headers: [.contentType: "application/json", .authorization: validKey],
                body: try requestBuffer()
            )
            XCTAssertEqual(resp.status, .unauthorized,
                           "Header without 'Bearer ' prefix must return 401")
        }
    }

    // MARK: Bearer with empty token → 401

    func testBearerEmptyToken_returns401() async throws {
        let app = Application(router: makeRouter(keys: [validKey]))

        try await app.test(.router) { client in
            let resp = try await client.execute(
                uri: "/v1/chat/completions",
                method: .post,
                headers: [.contentType: "application/json", .authorization: "Bearer "],
                body: try requestBuffer()
            )
            XCTAssertEqual(resp.status, .unauthorized,
                           "'Bearer ' with no token must return 401")
        }
    }

    // MARK: Correct key → 200

    func testValidKey_returns200() async throws {
        let app = Application(router: makeRouter(keys: [validKey]))

        try await app.test(.router) { client in
            let resp = try await client.execute(
                uri: "/v1/chat/completions",
                method: .post,
                headers: [.contentType: "application/json", .authorization: "Bearer \(validKey)"],
                body: try requestBuffer()
            )
            XCTAssertEqual(resp.status, .ok, "Valid API key must return 200")
        }
    }

    // MARK: Any key in the set is accepted

    func testSecondValidKey_returns200() async throws {
        let app = Application(router: makeRouter(keys: [validKey, anotherValidKey]))

        try await app.test(.router) { client in
            let resp = try await client.execute(
                uri: "/v1/chat/completions",
                method: .post,
                headers: [.contentType: "application/json",
                          .authorization: "Bearer \(anotherValidKey)"],
                body: try requestBuffer()
            )
            XCTAssertEqual(resp.status, .ok, "Any key in the allowed set must be accepted")
        }
    }

    // MARK: Key-file loading — parses keys, skips comments and blank lines

    func testLoadFromFile_parsesKeysAndSkipsComments() throws {
        let content = """
        # This is a comment
        sk-key-one

        sk-key-two
        # Another comment
        sk-key-three
        """
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("test-keys-\(UUID().uuidString).txt")
        try content.write(to: url, atomically: true, encoding: .utf8)
        defer { try? FileManager.default.removeItem(at: url) }

        // Should not throw — 3 valid keys were found
        _ = try BearerAuthMiddleware<BasicRequestContext>.loadFromFile(at: url.path)
    }

    // MARK: Key-file with no valid keys throws

    func testLoadFromFile_emptyFile_throws() throws {
        let content = "# only comments\n\n  \n"
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("empty-keys-\(UUID().uuidString).txt")
        try content.write(to: url, atomically: true, encoding: .utf8)
        defer { try? FileManager.default.removeItem(at: url) }

        XCTAssertThrowsError(
            try BearerAuthMiddleware<BasicRequestContext>.loadFromFile(at: url.path)
        ) { error in
            guard case BearerAuthError.noKeysInFile = error else {
                return XCTFail("Expected BearerAuthError.noKeysInFile, got \(error)")
            }
        }
    }

}

// MARK: - Assertion helpers

private func assertJSONError(_ resp: TestResponse) {
    XCTAssertEqual(resp.headers[.contentType], "application/json",
                   "401 responses must be JSON")
    let body = String(buffer: resp.body)
    XCTAssertTrue(body.contains("error"), "401 body must contain an 'error' key")
}
