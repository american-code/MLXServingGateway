import Foundation
import Hummingbird

public typealias ChatHandler = @Sendable (ChatCompletionRequest) async throws -> ChatCompletionResponse

public struct ChatRouter: Sendable {
    let handler: ChatHandler

    public init(handler: @escaping ChatHandler) {
        self.handler = handler
    }

    public func addRoutes(to router: Router<some RequestContext>) {
        router.post("/v1/chat/completions", use: completions)
    }

    @Sendable
    func completions(_ request: Request, context: some RequestContext) async throws -> Response {
        let body = try await request.body.collect(upTo: 10 * 1024 * 1024)
        let decoder = JSONDecoder()
        let chatRequest: ChatCompletionRequest
        do {
            chatRequest = try decoder.decode(ChatCompletionRequest.self, from: body)
        } catch {
            let apiError = APIError(message: "Invalid request body: \(error.localizedDescription)", type: "invalid_request_error")
            let data = try JSONEncoder().encode(apiError)
            return Response(
                status: .badRequest,
                headers: [.contentType: "application/json"],
                body: .init(byteBuffer: .init(bytes: data))
            )
        }

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
}
