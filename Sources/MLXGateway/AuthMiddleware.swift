import Foundation
import Hummingbird

/// Validates `Authorization: Bearer <token>` against an allowlist of keys.
/// Returns a 401 JSON response for missing or unrecognised credentials.
public struct BearerAuthMiddleware<Context: RequestContext>: MiddlewareProtocol {
    private let validKeys: Set<String>

    public init(keys: Set<String>) {
        self.validKeys = keys
    }

    public func handle(
        _ request: Request,
        context: Context,
        next: (Request, Context) async throws -> Response
    ) async throws -> Response {
        guard let token = bearerToken(from: request) else {
            return jsonError(status: .unauthorized, message: "Missing or malformed Authorization header")
        }
        guard validKeys.contains(token) else {
            return jsonError(status: .unauthorized, message: "Invalid API key")
        }
        return try await next(request, context)
    }

    // MARK: - Key-file loading

    /// Loads keys from a text file — one key per line; `#`-prefixed lines are comments.
    public static func loadFromFile(at path: String) throws -> BearerAuthMiddleware {
        let content = try String(contentsOfFile: path, encoding: .utf8)
        let keys = Set(
            content.components(separatedBy: .newlines)
                .map { $0.trimmingCharacters(in: .whitespaces) }
                .filter { !$0.isEmpty && !$0.hasPrefix("#") }
        )
        guard !keys.isEmpty else {
            throw BearerAuthError.noKeysInFile(path: path)
        }
        return BearerAuthMiddleware(keys: keys)
    }

    // MARK: - Helpers

    private func bearerToken(from request: Request) -> String? {
        guard let header = request.headers[.authorization],
              header.hasPrefix("Bearer ") else { return nil }
        let token = String(header.dropFirst("Bearer ".count))
            .trimmingCharacters(in: .whitespaces)
        return token.isEmpty ? nil : token
    }

    private func jsonError(status: HTTPResponse.Status, message: String) -> Response {
        let body = APIError(message: message, type: "invalid_request_error", code: "invalid_api_key")
        let data = (try? JSONEncoder().encode(body)) ?? Data()
        return Response(
            status: status,
            headers: [.contentType: "application/json"],
            body: .init(byteBuffer: .init(bytes: data))
        )
    }
}

public enum BearerAuthError: Error, Sendable {
    case noKeysInFile(path: String)
}
