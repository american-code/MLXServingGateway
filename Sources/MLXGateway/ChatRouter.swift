import Foundation
import Hummingbird

public typealias ChatHandler = @Sendable (ChatCompletionRequest) async throws -> ChatCompletionResponse
public typealias StreamHandler = @Sendable (ChatCompletionRequest) async throws -> AsyncThrowingStream<String, Error>

public struct ChatRouter: Sendable {
    let handler: ChatHandler
    let streamHandler: StreamHandler?
    /// Per-request deadline in seconds. `nil` disables the timeout.
    let timeoutSeconds: Double?

    public init(
        handler: @escaping ChatHandler,
        streamHandler: StreamHandler? = nil,
        timeoutSeconds: Double? = nil
    ) {
        self.handler = handler
        self.streamHandler = streamHandler
        self.timeoutSeconds = timeoutSeconds
    }

    public func addRoutes(to router: Router<some RequestContext>) {
        router.post("/v1/chat/completions", use: completions)
    }

    @Sendable
    func completions(_ request: Request, context: some RequestContext) async throws -> Response {
        let body = try await request.body.collect(upTo: 10 * 1024 * 1024)
        let chatRequest: ChatCompletionRequest
        do {
            chatRequest = try JSONDecoder().decode(ChatCompletionRequest.self, from: body)
        } catch {
            return errorResponse(
                status: .badRequest,
                message: "Invalid request body: \(error.localizedDescription)"
            )
        }

        do {
            if chatRequest.stream == true, let streamHandler {
                return try await sseResponse(for: chatRequest, using: streamHandler)
            }
            return try await jsonResponse(for: chatRequest)
        } catch is GatewayTimeoutError {
            return errorResponse(status: .gatewayTimeout, message: "Request timed out")
        }
    }

    // MARK: - Response builders

    private func jsonResponse(for chatRequest: ChatCompletionRequest) async throws -> Response {
        let completion: ChatCompletionResponse
        if let seconds = timeoutSeconds {
            completion = try await withDeadline(seconds: seconds) { [handler] in
                try await handler(chatRequest)
            }
        } else {
            completion = try await handler(chatRequest)
        }
        let encoder = JSONEncoder()
        encoder.outputFormatting = .sortedKeys
        let data = try encoder.encode(completion)
        return Response(
            status: .ok,
            headers: [.contentType: "application/json"],
            body: .init(byteBuffer: .init(bytes: data))
        )
    }

    private func sseResponse(
        for chatRequest: ChatCompletionRequest,
        using streamHandler: @escaping StreamHandler
    ) async throws -> Response {
        let id = "chatcmpl-\(UUID().uuidString)"
        let created = Int(Date().timeIntervalSince1970)
        let model = chatRequest.model

        let tokenStream: AsyncThrowingStream<String, Error>
        if let seconds = timeoutSeconds {
            tokenStream = try await withDeadline(seconds: seconds) { [streamHandler] in
                try await streamHandler(chatRequest)
            }
        } else {
            tokenStream = try await streamHandler(chatRequest)
        }

        let responseBody = ResponseBody { writer in
            let encoder = JSONEncoder()
            encoder.outputFormatting = .sortedKeys

            func event(_ chunk: ChatCompletionChunk) throws -> ByteBuffer {
                let json = try encoder.encode(chunk)
                return ByteBuffer(string: "data: \(String(decoding: json, as: UTF8.self))\n\n")
            }

            try await writer.write(event(ChatCompletionChunk(
                id: id, created: created, model: model,
                choices: [StreamChoice(index: 0, delta: DeltaMessage(role: .assistant))]
            )))

            for try await token in tokenStream {
                try await writer.write(event(ChatCompletionChunk(
                    id: id, created: created, model: model,
                    choices: [StreamChoice(index: 0, delta: DeltaMessage(content: token))]
                )))
            }

            try await writer.write(event(ChatCompletionChunk(
                id: id, created: created, model: model,
                choices: [StreamChoice(index: 0, delta: DeltaMessage(), finishReason: .stop)]
            )))
            try await writer.write(ByteBuffer(string: "data: [DONE]\n\n"))
            try await writer.finish(nil)
        }

        return Response(
            status: .ok,
            headers: [.contentType: "text/event-stream", .cacheControl: "no-cache"],
            body: responseBody
        )
    }

    private func errorResponse(status: HTTPResponse.Status, message: String) -> Response {
        let err = APIError(message: message, type: "server_error")
        let data = (try? JSONEncoder().encode(err)) ?? Data()
        return Response(
            status: status,
            headers: [.contentType: "application/json"],
            body: .init(byteBuffer: .init(bytes: data))
        )
    }

    // MARK: - Timeout helper

    /// Races `operation` against a sleep of `seconds`. Throws `GatewayTimeoutError` if the
    /// sleep wins. Cancels the losing task immediately after the winner finishes.
    private func withDeadline<T: Sendable>(
        seconds: Double,
        operation: @Sendable @escaping () async throws -> T
    ) async throws -> T {
        try await withThrowingTaskGroup(of: T.self) { group in
            group.addTask { try await operation() }
            group.addTask {
                try await Task.sleep(nanoseconds: UInt64(seconds * 1_000_000_000))
                throw GatewayTimeoutError()
            }
            let result = try await group.next()!
            group.cancelAll()
            return result
        }
    }
}
