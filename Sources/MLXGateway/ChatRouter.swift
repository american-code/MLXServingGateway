import Foundation
import Hummingbird

public typealias ChatHandler = @Sendable (ChatCompletionRequest) async throws -> ChatCompletionResponse
public typealias StreamHandler = @Sendable (ChatCompletionRequest) async throws -> AsyncThrowingStream<String, Error>

public struct ChatRouter: Sendable {
    let handler: ChatHandler
    let streamHandler: StreamHandler?

    public init(handler: @escaping ChatHandler, streamHandler: StreamHandler? = nil) {
        self.handler = handler
        self.streamHandler = streamHandler
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
            let apiError = APIError(message: "Invalid request body: \(error.localizedDescription)", type: "invalid_request_error")
            let data = try JSONEncoder().encode(apiError)
            return Response(
                status: .badRequest,
                headers: [.contentType: "application/json"],
                body: .init(byteBuffer: .init(bytes: data))
            )
        }

        if chatRequest.stream == true, let streamHandler {
            return try await sseResponse(for: chatRequest, using: streamHandler)
        }
        return try await jsonResponse(for: chatRequest)
    }

    private func jsonResponse(for chatRequest: ChatCompletionRequest) async throws -> Response {
        let completion = try await handler(chatRequest)
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
        using streamHandler: StreamHandler
    ) async throws -> Response {
        let id = "chatcmpl-\(UUID().uuidString)"
        let created = Int(Date().timeIntervalSince1970)
        let model = chatRequest.model
        let tokenStream = try await streamHandler(chatRequest)

        let responseBody = ResponseBody { writer in
            let encoder = JSONEncoder()
            encoder.outputFormatting = .sortedKeys

            func event(_ chunk: ChatCompletionChunk) throws -> ByteBuffer {
                let json = try encoder.encode(chunk)
                return ByteBuffer(string: "data: \(String(decoding: json, as: UTF8.self))\n\n")
            }

            // First chunk carries the role
            try await writer.write(event(ChatCompletionChunk(
                id: id, created: created, model: model,
                choices: [StreamChoice(index: 0, delta: DeltaMessage(role: .assistant))]
            )))

            // One chunk per token
            for try await token in tokenStream {
                try await writer.write(event(ChatCompletionChunk(
                    id: id, created: created, model: model,
                    choices: [StreamChoice(index: 0, delta: DeltaMessage(content: token))]
                )))
            }

            // Final chunk signals completion
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
}
