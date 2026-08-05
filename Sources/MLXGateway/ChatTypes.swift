import Foundation

// MARK: - Request

public struct ChatCompletionRequest: Codable, Sendable {
    public let model: String
    public let messages: [ChatMessage]
    public let temperature: Double?
    public let topP: Double?
    public let maxTokens: Int?
    public let stream: Bool?
    public let stop: StopSequence?
    public let presencePenalty: Double?
    public let frequencyPenalty: Double?
    public let seed: Int?
    public let user: String?

    public init(
        model: String,
        messages: [ChatMessage],
        temperature: Double? = nil,
        topP: Double? = nil,
        maxTokens: Int? = nil,
        stream: Bool? = nil,
        stop: StopSequence? = nil,
        presencePenalty: Double? = nil,
        frequencyPenalty: Double? = nil,
        seed: Int? = nil,
        user: String? = nil
    ) {
        self.model = model
        self.messages = messages
        self.temperature = temperature
        self.topP = topP
        self.maxTokens = maxTokens
        self.stream = stream
        self.stop = stop
        self.presencePenalty = presencePenalty
        self.frequencyPenalty = frequencyPenalty
        self.seed = seed
        self.user = user
    }

    enum CodingKeys: String, CodingKey {
        case model, messages, temperature, stop, stream, seed, user
        case topP = "top_p"
        case maxTokens = "max_tokens"
        case presencePenalty = "presence_penalty"
        case frequencyPenalty = "frequency_penalty"
    }
}

public enum StopSequence: Codable, Sendable {
    case single(String)
    case multiple([String])

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let s = try? container.decode(String.self) {
            self = .single(s)
        } else {
            self = .multiple(try container.decode([String].self))
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .single(let s): try container.encode(s)
        case .multiple(let a): try container.encode(a)
        }
    }

    public var strings: [String] {
        switch self {
        case .single(let s): return [s]
        case .multiple(let a): return a
        }
    }
}

// MARK: - Streaming chunks

/// A single item yielded by a streaming generation handler.
public enum StreamChunk: Sendable {
    case token(String)
    case done(FinishReason)
}

// MARK: - Message

public struct ChatMessage: Codable, Sendable {
    public let role: Role
    public let content: MessageContent

    public init(role: Role, content: String) {
        self.role = role
        self.content = .text(content)
    }

    public init(role: Role, content: MessageContent) {
        self.role = role
        self.content = content
    }
}

public enum Role: String, Codable, Sendable {
    case system
    case user
    case assistant
    case tool
}

public enum MessageContent: Codable, Sendable {
    case text(String)
    case parts([ContentPart])

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let s = try? container.decode(String.self) {
            self = .text(s)
        } else {
            self = .parts(try container.decode([ContentPart].self))
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .text(let s): try container.encode(s)
        case .parts(let p): try container.encode(p)
        }
    }

    public var textContent: String {
        switch self {
        case .text(let s): return s
        case .parts(let parts): return parts.compactMap(\.text).joined()
        }
    }
}

public struct ContentPart: Codable, Sendable {
    public let type: String
    public let text: String?

    public init(type: String, text: String?) {
        self.type = type
        self.text = text
    }
}

// MARK: - Response

public struct ChatCompletionResponse: Codable, Sendable {
    public let id: String
    public let object: String
    public let created: Int
    public let model: String
    public let choices: [Choice]
    public let usage: Usage?
    public let systemFingerprint: String?

    public init(
        id: String,
        object: String = "chat.completion",
        created: Int,
        model: String,
        choices: [Choice],
        usage: Usage? = nil,
        systemFingerprint: String? = nil
    ) {
        self.id = id
        self.object = object
        self.created = created
        self.model = model
        self.choices = choices
        self.usage = usage
        self.systemFingerprint = systemFingerprint
    }

    enum CodingKeys: String, CodingKey {
        case id, object, created, model, choices, usage
        case systemFingerprint = "system_fingerprint"
    }
}

public struct Choice: Codable, Sendable {
    public let index: Int
    public let message: ChatMessage
    public let finishReason: FinishReason?
    public let logprobs: String?

    public init(index: Int, message: ChatMessage, finishReason: FinishReason? = .stop, logprobs: String? = nil) {
        self.index = index
        self.message = message
        self.finishReason = finishReason
        self.logprobs = logprobs
    }

    enum CodingKeys: String, CodingKey {
        case index, message, logprobs
        case finishReason = "finish_reason"
    }
}

public enum FinishReason: String, Codable, Sendable {
    case stop
    case length
    case contentFilter = "content_filter"
    case toolCalls = "tool_calls"
}

public struct Usage: Codable, Sendable {
    public let promptTokens: Int
    public let completionTokens: Int
    public let totalTokens: Int

    public init(promptTokens: Int, completionTokens: Int) {
        self.promptTokens = promptTokens
        self.completionTokens = completionTokens
        self.totalTokens = promptTokens + completionTokens
    }

    enum CodingKeys: String, CodingKey {
        case promptTokens = "prompt_tokens"
        case completionTokens = "completion_tokens"
        case totalTokens = "total_tokens"
    }
}

// MARK: - Streaming (SSE)

public struct ChatCompletionChunk: Codable, Sendable {
    public let id: String
    public let object: String
    public let created: Int
    public let model: String
    public let choices: [StreamChoice]

    public init(id: String, created: Int, model: String, choices: [StreamChoice]) {
        self.id = id
        self.object = "chat.completion.chunk"
        self.created = created
        self.model = model
        self.choices = choices
    }
}

public struct StreamChoice: Codable, Sendable {
    public let index: Int
    public let delta: DeltaMessage
    public let finishReason: FinishReason?

    public init(index: Int, delta: DeltaMessage, finishReason: FinishReason? = nil) {
        self.index = index
        self.delta = delta
        self.finishReason = finishReason
    }

    enum CodingKeys: String, CodingKey {
        case index, delta
        case finishReason = "finish_reason"
    }
}

public struct DeltaMessage: Codable, Sendable {
    public let role: Role?
    public let content: String?

    public init(role: Role? = nil, content: String? = nil) {
        self.role = role
        self.content = content
    }
}

// MARK: - Models list (GET /v1/models)

public struct ModelListResponse: Codable, Sendable {
    public let object: String
    public let data: [ModelInfo]

    public init(data: [ModelInfo]) {
        self.object = "list"
        self.data = data
    }
}

public struct ModelInfo: Codable, Sendable {
    public let id: String
    public let object: String
    public let created: Int
    public let ownedBy: String

    public init(id: String, created: Int = 0, ownedBy: String = "local") {
        self.id = id
        self.object = "model"
        self.created = created
        self.ownedBy = ownedBy
    }

    enum CodingKeys: String, CodingKey {
        case id, object, created
        case ownedBy = "owned_by"
    }
}

// MARK: - Gateway errors

/// Thrown (and caught inside `ChatRouter`) when a request exceeds the configured deadline.
public struct GatewayTimeoutError: Error, Sendable {}

// MARK: - Error

public struct APIError: Codable, Sendable {
    public let error: ErrorDetail

    public init(message: String, type: String = "server_error", code: String? = nil) {
        self.error = ErrorDetail(message: message, type: type, code: code)
    }
}

public struct ErrorDetail: Codable, Sendable {
    public let message: String
    public let type: String
    public let param: String?
    public let code: String?

    public init(message: String, type: String, param: String? = nil, code: String? = nil) {
        self.message = message
        self.type = type
        self.param = param
        self.code = code
    }
}
